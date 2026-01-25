import logging

import numpy as np
import onnx_ir as ir

from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._gptq import _gptq_quantize
from onnx_quantize.core._rtn import _rtn_quantize


logger = logging.getLogger(__name__)


def _resolve_group_size(w: ir.Value, group_size: int) -> int:
    in_channels = w.const_value.numpy().shape[0]

    log_msg = f"Adjusting group size from {group_size} to {in_channels} for weight '{w.name}'"

    if group_size:
        if group_size > in_channels:
            logger.debug(log_msg + f" as it exceeds the number of input channels {in_channels}.")
            group_size = in_channels

        if in_channels % group_size != 0:
            logger.debug(
                log_msg + f" as it does not divide the number of input channels {in_channels}."
            )
            group_size = in_channels

    return group_size


def _quantize_weights_gptq(w: ir.Value, inputs: np.ndarray, qconfig: QConfig):
    w_q, w_scale, w_zero_point = _gptq_quantize(
        w.const_value.numpy(),
        inputs,
        quant_type=qconfig.weights.dtype,
        strategy=qconfig.weights.strategy,
        is_symmetric=qconfig.weights.symmetric,
        reduce_range=qconfig.weights.reduce_range,
        clip_ratio=qconfig.weights.clip_ratio,
        block_size=qconfig.weights.algorithm.block_size,
        percdamp=qconfig.weights.algorithm.percdamp,
        group_size=qconfig.weights.group_size,
        actorder=qconfig.weights.algorithm.actorder,
        mse=qconfig.weights.mse,
        scale_dtype=qconfig.weights.scale_dtype,
        zp_dtype=qconfig.weights.zp_dtype,
    )

    # Cast to scale dtype
    w_scale = w_scale.astype(qconfig.weights.scale_dtype)

    return w_q, w_scale, w_zero_point


def _quantize_weights_rtn(w: ir.Value, qconfig: QConfig):
    w_q, w_scale, w_zero_point = _rtn_quantize(
        w.const_value.numpy(),
        qconfig.weights.dtype,
        strategy=qconfig.weights.strategy,
        group_size=qconfig.weights.group_size,
        is_symmetric=qconfig.weights.symmetric,
        reduce_range=qconfig.weights.reduce_range,
        clip_ratio=qconfig.weights.clip_ratio,
        mse=qconfig.weights.mse,
        scale_dtype=qconfig.weights.scale_dtype,
        zp_dtype=qconfig.weights.zp_dtype,
    )

    return w_q, w_scale, w_zero_point

    return w_q, w_scale, w_zero_point


def is_matmul_nbits_compatible(qconfig: QConfig, name: str = "") -> bool:
    weights_only = qconfig.input_activations is None and qconfig.output_activations is None
    log_msg = f"Found uncompatibility for MatMulNBits in {name}: "

    # TODO: it is possible to do it also with QDQ
    if not weights_only:
        logger.debug(log_msg + "It only supports weight-only quantization.")
        return False

    if qconfig.weights.dtype not in {QuantType.QUInt4, QuantType.QUInt8}:
        logger.debug(
            log_msg
            + f"It only supports uint4 and uint8 weight types. Found: {qconfig.weights.dtype}"
        )
        return False

    if qconfig.weights.strategy != QuantizationStrategy.GROUP:
        logger.debug(
            log_msg
            + "It only supports 'group' quantization strategy. Found: "
            + str(qconfig.weights.strategy)
        )
        return False

    # group_size should be greater than 16 and should be power of 2
    group_size = qconfig.weights.group_size
    if group_size != -1 and (group_size < 16 or (group_size & (group_size - 1)) != 0):
        logger.debug(log_msg + "group_size should be a power of 2 greater than or equal to 16.")
        return False

    return True


def _prepare_for_matmul_nbits(
    w_q: np.ndarray, w_scale: np.ndarray, w_zero_point: np.ndarray, qconfig: QConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_channels, out_channels = w_q.shape
    num_bits = qconfig.weights.dtype.bitwidth

    # Assert that in_channels is divisible by group_size
    assert in_channels % qconfig.weights.group_size == 0
    num_blocks = in_channels // qconfig.weights.group_size

    # Prepare weights
    w_q = np.reshape(w_q.T, (-1, qconfig.weights.group_size))
    blob_size = qconfig.weights.group_size * num_bits // 8

    # Pack weights for 4-bit quantization
    if num_bits == 4:
        packed = np.zeros((w_q.shape[0], blob_size), dtype=np.uint8)
        pack_weight_pair = (w_q[:, ::2]) | (w_q[:, 1::2] << 4)
        packed[:, :] = pack_weight_pair[:, :blob_size]
        # TODO: optimize this
        w_q = packed

    w_q = np.reshape(w_q, (-1, num_blocks, blob_size))

    # Reshape scale to (out_channels, num_blocks)
    w_scale = w_scale.reshape(-1, num_blocks)

    # Prepare zero point
    packed_zp = w_zero_point

    # For the case where num_blocks = 1, we don't pack the zero points
    if num_bits == 4 and num_blocks > 1:
        # For 4-bit case, the default zeros is 0x8. So it is 0x88 = 136
        # if we fill lower/higher 4 bits with 0x8.
        packed_zp = np.full((w_zero_point.shape[0] + 1) // 2, 136, dtype="uint8")

        # create an index array
        idx = np.arange(w_zero_point.shape[0] // num_blocks * num_blocks).reshape(-1)

        # separate odd and even indices
        even_idx = idx[::2]
        odd_idx = idx[1::2]
        # vectorized operation for even and odd indices
        packed_zp[even_idx // 2] = (packed_zp[even_idx // 2] & 0xF0) | w_zero_point[
            even_idx
        ].ravel()
        packed_zp[odd_idx // 2] = (packed_zp[odd_idx // 2] & 0x0F) | (
            w_zero_point[odd_idx].ravel() << 4
        )
    packed_zp = np.reshape(packed_zp, (out_channels, -1)).astype(np.uint8)

    return w_q, w_scale, packed_zp


def quantize_weights(
    op: ir.tape.Tape,
    w: ir.Value,
    qconfig: QConfig,
    out: ir.Value | None = None,
    is_matmul_nbits_compatible: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    if isinstance(qconfig.weights.algorithm, GPTQConfig):
        assert out is not None, "Output value is required for GPTQ quantization."
        node = out.producer()
        assert "input" in node.meta, "GPTQ requires calibration data in node meta."
        w_q, w_scale, w_zero_point = _quantize_weights_gptq(w, node.meta["input"], qconfig)
    else:
        w_q, w_scale, w_zero_point = _quantize_weights_rtn(w, qconfig)

    # Prepare quantized weights, scales, and zero points if qconfig is compatible with MatMulNBits
    if is_matmul_nbits_compatible:
        w_q, w_scale, w_zero_point = _prepare_for_matmul_nbits(w_q, w_scale, w_zero_point, qconfig)

    # Create ONNX tensors from quantized weights
    w_q = op.initializer(ir.tensor(w_q), name=w.name)
    w_scale = op.initializer(ir.tensor(w_scale), name=f"{w.name}/scale")
    w_zero_point = op.initializer(ir.tensor(w_zero_point), name=f"{w.name}/zero_point")

    return w_q, w_scale, w_zero_point

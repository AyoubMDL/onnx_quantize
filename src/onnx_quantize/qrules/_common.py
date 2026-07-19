import logging

import numpy as np
import onnx_ir as ir

from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy


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
    if num_bits == 4 and num_blocks > 1 and qconfig.weights.zp_dtype != w_scale.dtype:
        # Pack per row so each output channel's zero points stay contiguous,
        # padding to an even count with the default 0x8 nibble when num_blocks is odd.
        # Example
        # shape (4, 5):
        # [[a0, a1, a2, a3, a4],
        #  [b0, b1, b2, b3, b4],
        #  [c0, c1, c2, c3, c4],
        #  [d0, d1, d2, d3, d4]]
        zp = w_zero_point.reshape(out_channels, num_blocks).astype(np.uint8)
        if num_blocks % 2 == 1:
            # shape (4, 6):
            # [[a0, a1, a2, a3, a4, 0x8],
            #  [b0, b1, b2, b3, b4, 0x8],
            #  ...]
            pad = np.full((out_channels, 1), 0x8, dtype=np.uint8)
            zp = np.concatenate([zp, pad], axis=1)

        # packed_zp shape (4, 3):
        # [[a1|a0, a3|a2, 0x8|a4],
        #  [b1|b0, b3|b2, 0x8|b4],
        #  ...]
        packed_zp = (zp[:, ::2] & 0x0F) | ((zp[:, 1::2] & 0x0F) << 4)

    zp_dtype = np.uint8 if qconfig.weights.zp_dtype != w_scale.dtype else qconfig.weights.zp_dtype
    packed_zp = np.reshape(packed_zp, (out_channels, -1)).astype(zp_dtype)

    return w_q, w_scale, packed_zp


def quantize_weights(
    op: ir.tape.Tape,
    w: ir.Value,
    qconfig: QConfig,
    out: ir.Value | None = None,
    is_matmul_nbits_compatible: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    w_q, w_scale, w_zero_point = qconfig.weights.algorithm.quantize_weights(w, qconfig, out=out)

    # Prepare quantized weights, scales, and zero points if qconfig is compatible with MatMulNBits
    if is_matmul_nbits_compatible:
        w_q, w_scale, w_zero_point = _prepare_for_matmul_nbits(w_q, w_scale, w_zero_point, qconfig)

    # Create ONNX tensors from quantized weights
    w_q = op.initializer(ir.tensor(w_q), name=w.name)
    w_scale = op.initializer(ir.tensor(w_scale), name=f"{w.name}/scale")
    w_zero_point = op.initializer(ir.tensor(w_zero_point), name=f"{w.name}/zero_point")

    return w_q, w_scale, w_zero_point

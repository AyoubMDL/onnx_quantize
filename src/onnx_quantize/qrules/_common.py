import numpy as np
import onnx_ir as ir

from onnx_quantize.core._gptq import _gptq_quantize
from onnx_quantize.core._qconfig import GPTQConfig, QConfig
from onnx_quantize.core._rtn import _quantize_array


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
        group_size=qconfig.weights.algorithm.group_size,
        actorder=qconfig.weights.algorithm.actorder,
        mse=qconfig.weights.mse,
    )

    # Post process qparams
    w_scale = np.squeeze(w_scale.astype(qconfig.weights.scale_dtype))
    w_zero_point = np.squeeze(w_zero_point)

    return w_q, w_scale, w_zero_point


def _quantize_weights_rtn(w: ir.Value, qconfig: QConfig):
    w_q, w_scale, w_zero_point = _quantize_array(
        w.const_value.numpy(),
        qconfig.weights.dtype,
        strategy=qconfig.weights.strategy,
        group_size=qconfig.weights.group_size,
        is_symmetric=qconfig.weights.symmetric,
        reduce_range=qconfig.weights.reduce_range,
        clip_ratio=qconfig.weights.clip_ratio,
        mse=qconfig.weights.mse,
    )

    # Cast to scale dtype
    w_scale = w_scale.astype(qconfig.weights.scale_dtype)

    return w_q, w_scale, w_zero_point


def quantize_weights(op: ir.tape.Tape, w: ir.Value, qconfig: QConfig, out: ir.Value | None = None):
    if isinstance(qconfig.weights.algorithm, GPTQConfig):
        assert out is not None, "Output value is required for GPTQ quantization."
        node = out.producer()
        assert "input" in node.meta, "GPTQ requires calibration data in node meta."
        w_q, w_scale, w_zero_point = _quantize_weights_gptq(w, node.meta["input"], qconfig)
    else:
        w_q, w_scale, w_zero_point = _quantize_weights_rtn(w, qconfig)

    # Create ONNX tensors from quantized weights
    w_q = op.initializer(ir.tensor(w_q), name=w.name)
    w_scale = op.initializer(ir.tensor(w_scale), name=f"{w.name}/scale")
    w_zero_point = op.initializer(ir.tensor(w_zero_point), name=f"{w.name}/zero_point")

    return w_q, w_scale, w_zero_point

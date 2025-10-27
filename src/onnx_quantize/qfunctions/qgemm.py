import onnx_ir as ir
from onnxscript import opset20 as op
from onnxscript import script

from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction
@script(opset=QUANT_OPSET)
def QGemmStatic8bits(X, W, B, x_scale, w_scale, x_zero_point, w_zero_point):
    """Static Quantized Gemm using ONNX ops."""
    # Quantize the inputs
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Int MatMul (W is already quantized)
    out_matmul = op.MatMulInteger(x_quantized, W, x_zero_point, w_zero_point)

    # Bias already in int32
    out_bias = op.Add(out_matmul, B)
    dequantized_matmul = op.DequantizeLinear(out_bias, x_scale * w_scale)

    return dequantized_matmul


@register_qfunction
@script(opset=QUANT_OPSET)
def _quantize_bias(B, x_scale, w_scale):
    bias_scale = op.Mul(x_scale, w_scale)
    q_bias = op.Div(B, bias_scale)
    min_ = op.Cast(-(2**31), to=ir.DataType.INT32)
    max_ = op.Cast(2 ** (31) - 1, to=ir.DataType.INT32)
    q_bias = op.Cast(q_bias, to=ir.DataType.INT32)
    q_bias = op.Clip(q_bias, min_, max_)
    return q_bias


@register_qfunction
@script(opset=QUANT_OPSET)
def QGemmDynamic8bits(X, W, B, w_scale, w_zero_point):
    """Dynamic Quantized Gemm using ONNX ops."""
    # Dynamicly quantize the inputs
    # TODO: Replace this with onnx ops to support int8 (now only supporting uint8)
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Int MatMul (W is already quantized)
    out_matmul = op.MatMulInteger(x_quantized, W, x_zero_point, w_zero_point)

    # Quantize bias
    bias_quantized = _quantize_bias(B, x_scale, w_scale)

    # Quantize bias
    out_bias = op.Add(out_matmul, bias_quantized)
    dequantized_matmul = op.DequantizeLinear(out_bias, x_scale * w_scale)

    return dequantized_matmul

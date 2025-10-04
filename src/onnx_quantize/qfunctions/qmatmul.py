from onnxscript import opset20 as op
from onnxscript import script

from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction
@script(opset=QUANT_OPSET)
def QMatMulStatic8bits(X, W, x_scale, w_scale, x_zero_point, w_zero_point):
    """Static Quantized MatMul using ONNX ops."""
    # Quantize the inputs
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Int MatMul (W is already quantized)
    out_matmul = op.MatMulInteger(x_quantized, W, x_zero_point, w_zero_point)
    dequantized_matmul = op.DequantizeLinear(out_matmul, x_scale * w_scale)

    return dequantized_matmul


@register_qfunction
@script(opset=QUANT_OPSET)
def QMatMulDynamic8bits(X, W, w_scale, w_zero_point):
    """Dynamic Quantized MatMul using ONNX ops."""
    # Dynamicly quantize the inputs
    # TODO: Replace this with onnx ops to support int8 (now only supporting uint8)
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Int MatMul (W is already quantized)
    out_matmul = op.MatMulInteger(x_quantized, W, x_zero_point, w_zero_point)
    dequantized_matmul = op.DequantizeLinear(out_matmul, x_scale * w_scale)

    return dequantized_matmul

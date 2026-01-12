from onnxscript import script

from onnx_quantize.opset import op
from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QLinearMatMul(X, W, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point):
    """Fully quantized MatMul with weight, input, and output activation quantization.

    Uses QLinearMatMul with all quantization parameters.
    """
    # Quantize input activation
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # QLinearMatMul with full quantization
    out = op.QLinearMatMul(
        x_quantized,
        x_scale,
        x_zero_point,
        W,
        w_scale,
        w_zero_point,
        out_scale,
        out_zero_point,
    )

    # Dequantize output activation
    out_dequantized = op.DequantizeLinear(out, out_scale, out_zero_point)

    return out_dequantized

import onnxscript
from onnxscript import script

from onnx_quantize.opset import op
from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


op_ms = onnxscript.values.Opset("com.microsoft", version=1)


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QLinearGemm(
    X,
    W,
    B,
    w_scale,
    w_zero_point,
    x_scale,
    x_zero_point,
    out_scale,
    out_zero_point,
):
    """Fully quantized Gemm with weight, bias, input, and output activation quantization.

    Uses QLinearGemm with all quantization parameters.
    """
    # Quantize input activation
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    matmul_out = op_ms.QGemm(
        x_quantized,
        x_scale,
        x_zero_point,
        W,
        w_scale,
        w_zero_point,
        B,
        out_scale,
        out_zero_point,
    )

    return op.DequantizeLinear(matmul_out, out_scale, out_zero_point)

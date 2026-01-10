"""QDQ-based quantized Gemm functions with flexible activation quantization support."""

import onnx_ir as ir
from onnxscript import script

from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy
from onnx_quantize.opset import op
from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightsOnlyQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point):
    """Weights-only quantized Gemm following QDQ pattern.

    Weights and bias are dequantized (bias is quantized per-tensor).
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Regular Gemm with dequantized weights and bias
    out = op.Gemm(X, dequantized_weights, dequantized_bias)

    return out


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightInputQDQ(
    X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, x_scale, x_zero_point
):
    """Quantized Gemm with weight and input activation quantization following QDQ pattern.

    Pattern: Q(input) -> DQ -> Gemm <- DQ <- Q(weight)
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Quantize input activation (Q)
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # Gemm with dequantized inputs
    out = op.Gemm(x_dequantized, dequantized_weights, dequantized_bias)

    return out


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightOutputQDQ(
    X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, out_scale, out_zero_point
):
    """Quantized Gemm with weight and output activation quantization following QDQ pattern.

    Pattern: Gemm -> Q(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Gemm with dequantized weights
    out = op.Gemm(X, dequantized_weights, dequantized_bias)

    # Quantize output activation (Q)
    out_quantized = op.QuantizeLinear(out, out_scale, out_zero_point)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightInputOutputQDQ(
    X,
    W,
    B,
    w_scale,
    w_zero_point,
    b_scale,
    b_zero_point,
    x_scale,
    x_zero_point,
    out_scale,
    out_zero_point,
):
    """Fully quantized Gemm with weight, input, and output activation quantization.

    Follows QDQ pattern.
    Pattern: Q(input) -> DQ -> Gemm <- DQ <- Q(weight) -> Q(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Quantize input activation (Q)
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # Gemm with dequantized inputs
    out = op.Gemm(x_dequantized, dequantized_weights, dequantized_bias)

    # Quantize output activation (Q)
    out_quantized = op.QuantizeLinear(out, out_scale, out_zero_point)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightDynamicInputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point):
    """Dynamic quantized Gemm with weight and dynamic input activation following QDQ pattern.

    Input activation is dynamically quantized at runtime.
    Pattern: DynamicQ(input) -> DQ -> Gemm <- DQ <- Q(weight)
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Dynamically quantize input activation at runtime
    # Returns quantized tensor, scale, and zero_point
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # Gemm with dequantized inputs
    out = op.Gemm(x_dequantized, dequantized_weights, dequantized_bias)

    return out


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightDynamicOutputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point):
    """Dynamic quantized Gemm with weight and dynamic output activation following QDQ pattern.

    Output activation is dynamically quantized at runtime.
    Pattern: Gemm <- DQ <- Q(weight) -> DynamicQ(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Gemm with dequantized weights
    out = op.Gemm(X, dequantized_weights, dequantized_bias)

    # Dynamically quantize output activation at runtime
    out_quantized, out_scale, out_zero_point = op.DynamicQuantizeLinear(out)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="Gemm")
@script(opset=QUANT_OPSET)
def QGemmWeightDynamicInputOutputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point):
    """Fully dynamic quantized Gemm with weight, dynamic input and output activations.

    Both input and output activations are dynamically quantized at runtime.
    Pattern: DynamicQ(input) -> DQ -> Gemm <- DQ <- Q(weight) -> DynamicQ(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dequantize bias (Q -> DQ, per-tensor)
    dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

    # Dynamically quantize input activation at runtime
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # Gemm with dequantized inputs
    out = op.Gemm(x_dequantized, dequantized_weights, dequantized_bias)

    # Dynamically quantize output activation at runtime
    out_quantized, out_scale, out_zero_point = op.DynamicQuantizeLinear(out)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


def _make_qgemm_weight_only_grouped(group_size):
    @register_qfunction(target_optype="Gemm")
    @script(opset=QUANT_OPSET)
    def QGemmWeightsOnlyGrouped(
        X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, original_transposed_shape
    ):
        # (in_channels, out_channels) -> (out_channels x num_groups, group_size)
        W = op.Reshape(op.Transpose(W, perm=[1, 0]), op.Constant(value_ints=[-1, group_size]))
        dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point, block_size=group_size)

        # Reshape back to original and transpose
        # (out_channels x num_groups, group_size) -> (in_channels, out_channels)
        dequantized_weights = op.Transpose(
            op.Reshape(dequantized_weights, original_transposed_shape),
            perm=[1, 0],
        )

        # Dequantize bias (Q -> DQ, per-tensor)
        dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

        return op.Gemm(X, dequantized_weights, dequantized_bias)

    return QGemmWeightsOnlyGrouped


def _make_qgemm_weight_only_grouped_4bits(group_size):
    @register_qfunction(target_optype="Gemm")
    @script(opset=QUANT_OPSET)
    def QGemmWeightsOnlyGrouped(
        X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, original_transposed_shape
    ):
        # Cast to INT8 as Ort Reshape doesn't support INT4/UINT4
        W = op.Cast(W, to=ir.DataType.INT8)
        w_zero_point = op.Cast(w_zero_point, to=ir.DataType.INT8)

        # (in_channels, out_channels) -> (out_channels x num_groups, group_size)
        W = op.Reshape(op.Transpose(W, perm=[1, 0]), op.Constant(value_ints=[-1, group_size]))
        dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point, block_size=group_size)

        # Reshape back to original and transpose
        # (out_channels x num_groups, group_size) -> (in_channels, out_channels)
        dequantized_weights = op.Transpose(
            op.Reshape(dequantized_weights, original_transposed_shape),
            perm=[1, 0],
        )

        # Dequantize bias (Q -> DQ, per-tensor)
        dequantized_bias = op.DequantizeLinear(B, b_scale, b_zero_point)

        return op.Gemm(X, dequantized_weights, dequantized_bias)

    return QGemmWeightsOnlyGrouped


def qgemm_qdq_factory(qconfig: QConfig):
    """Factory function to return the appropriate QDQ Gemm operation based on provided parameters.

    Args:
        qconfig (QConfig): The quantization configuration.

    Returns:
        str: The appropriate QGemm function name based on which parameters are provided.
    """
    has_input_quant = qconfig.input_activations is not None
    has_output_quant = qconfig.output_activations is not None
    is_static = (has_input_quant and qconfig.input_activations.is_static) or (
        has_output_quant and qconfig.output_activations.is_static
    )

    if qconfig.weights.strategy == QuantizationStrategy.GROUP:
        # This will register a new QFunction for this group size
        group_size = qconfig.weights.group_size
        if qconfig.weights.dtype.bitwidth == 4:
            # Special case for grouped 4bits as ort doesn't support Reshape with 4bits inputs
            return _make_qgemm_weight_only_grouped_4bits(group_size)
        else:
            return _make_qgemm_weight_only_grouped(group_size)

    if is_static:
        if has_input_quant and has_output_quant:
            # Full quantization: weight + input + output
            return QGemmWeightInputOutputQDQ
        elif has_input_quant:
            # Weight + input quantization
            return QGemmWeightInputQDQ
        elif has_output_quant:
            # Weight + output quantization
            return QGemmWeightOutputQDQ
        else:
            # Weights-only quantization
            return QGemmWeightsOnlyQDQ
    else:
        if has_input_quant and has_output_quant:
            # Full dynamic quantization: weight + dynamic input + dynamic output
            return QGemmWeightDynamicInputOutputQDQ
        elif has_input_quant:
            # Weight + dynamic input quantization
            return QGemmWeightDynamicInputQDQ
        elif has_output_quant:
            # Weight + dynamic output quantization
            return QGemmWeightDynamicOutputQDQ
        else:
            # Weights-only quantization
            return QGemmWeightsOnlyQDQ

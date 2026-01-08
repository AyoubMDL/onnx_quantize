import onnx_ir as ir
from onnxscript import script

from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy
from onnx_quantize.opset import op
from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightsOnlyQDQ(X, W, w_scale, w_zero_point):
    """Weights-only quantized MatMul following QDQ pattern.

    Only weights are dequantized.
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Regular MatMul with dequantized weights
    out = op.MatMul(X, dequantized_weights)

    return out


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightStaticInputQDQ(X, W, w_scale, w_zero_point, x_scale, x_zero_point):
    """Quantized MatMul with weight and input activation quantization following QDQ pattern.

    Pattern: Q(input) -> DQ -> MatMul <- DQ <- Q(weight)
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Quantize input activation (Q)
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # MatMul with dequantized inputs
    out = op.MatMul(x_dequantized, dequantized_weights)

    return out


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightStaticOutputQDQ(X, W, w_scale, w_zero_point, out_scale, out_zero_point):
    """Quantized MatMul with weight and output activation quantization following QDQ pattern.

    Pattern: MatMul -> Q(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # MatMul with dequantized weights
    out = op.MatMul(X, dequantized_weights)

    # Quantize output activation (Q)
    out_quantized = op.QuantizeLinear(out, out_scale, out_zero_point)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightStaticInputOutputQDQ(
    X, W, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point
):
    """Fully quantized MatMul with weight, input, and output activation quantization.

    Follows QDQ pattern.
    Pattern: Q(input) -> DQ -> MatMul <- DQ <- Q(weight) -> Q(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Quantize input activation (Q)
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # MatMul with dequantized inputs
    out = op.MatMul(x_dequantized, dequantized_weights)

    # Quantize output activation (Q)
    out_quantized = op.QuantizeLinear(out, out_scale, out_zero_point)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightDynamicInputQDQ(X, W, w_scale, w_zero_point):
    """Dynamic quantized MatMul with weight and dynamic input activation following QDQ pattern.

    Input activation is dynamically quantized at runtime.
    Pattern: DynamicQ(input) -> DQ -> MatMul <- DQ <- Q(weight)
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dynamically quantize input activation at runtime
    # Returns quantized tensor, scale, and zero_point
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # MatMul with dequantized inputs
    out = op.MatMul(x_dequantized, dequantized_weights)

    return out


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightDynamicOutputQDQ(X, W, w_scale, w_zero_point):
    """Dynamic quantized MatMul with weight and dynamic output activation following QDQ pattern.

    Output activation is dynamically quantized at runtime.
    Pattern: MatMul <- DQ <- Q(weight) -> DynamicQ(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # MatMul with dequantized weights
    out = op.MatMul(X, dequantized_weights)

    # Dynamically quantize output activation at runtime
    out_quantized, out_scale, out_zero_point = op.DynamicQuantizeLinear(out)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightDynamicInputOutputQDQ(X, W, w_scale, w_zero_point):
    """Fully dynamic quantized MatMul with weight, dynamic input and output activations.

    Both input and output activations are dynamically quantized at runtime.
    Pattern: DynamicQ(input) -> DQ -> MatMul <- DQ <- Q(weight) -> DynamicQ(output) -> DQ
    """
    # Dequantize weights (Q -> DQ)
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)

    # Dynamically quantize input activation at runtime
    x_quantized, x_scale, x_zero_point = op.DynamicQuantizeLinear(X)

    # Dequantize input activation (DQ)
    x_dequantized = op.DequantizeLinear(x_quantized, x_scale, x_zero_point)

    # MatMul with dequantized inputs
    out = op.MatMul(x_dequantized, dequantized_weights)

    # Dynamically quantize output activation at runtime
    out_quantized, out_scale, out_zero_point = op.DynamicQuantizeLinear(out)

    # Dequantize output activation (DQ)
    out_dequantized = op.DequantizeLinear(out_quantized, out_scale, out_zero_point)

    return out_dequantized


def _make_qmatmul_weight_only_grouped(group_size):
    @register_qfunction(target_optype="MatMul")
    @script(opset=QUANT_OPSET)
    def QMatMulWeightsOnlyGrouped(X, W, w_scale, w_zero_point, original_transposed_shape):
        # (in_channels, out_channels) -> (out_channels x num_groups, group_size)
        W = op.Reshape(op.Transpose(W, perm=[1, 0]), op.Constant(value_ints=[-1, group_size]))
        dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point, block_size=group_size)

        # Reshape back to original and transpose
        # (out_channels x num_groups, group_size) -> (in_channels, out_channels)
        dequantized_weights = op.Transpose(
            op.Reshape(dequantized_weights, original_transposed_shape),
            perm=[1, 0],
        )
        return op.MatMul(X, dequantized_weights)

    return QMatMulWeightsOnlyGrouped


def _make_qmatmul_weight_only_grouped_4bits(group_size):
    @register_qfunction(target_optype="MatMul")
    @script(opset=QUANT_OPSET)
    def QMatMulWeightsOnlyGrouped(X, W, w_scale, w_zero_point, original_transposed_shape):
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
        return op.MatMul(X, dequantized_weights)

    return QMatMulWeightsOnlyGrouped


def qmatmul_qdq_factory(qconfig: QConfig):
    """Factory function to return the appropriate QDQ MatMul operation based on provided parameters.

    Args:
        qconfig (QConfig): The quantization configuration.

    Returns:
        The appropriate QMatMul function result based on which parameters are provided.
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
            return _make_qmatmul_weight_only_grouped_4bits(group_size)
        else:
            return _make_qmatmul_weight_only_grouped(group_size)

    if is_static:
        if has_input_quant and has_output_quant:
            # Full quantization: weight + input + output
            return QMatMulWeightStaticInputOutputQDQ
        elif has_input_quant:
            # Weight + input quantization
            return QMatMulWeightStaticInputQDQ
        elif has_output_quant:
            # Weight + output quantization
            return QMatMulWeightStaticOutputQDQ
        else:
            # Weights-only quantization
            return QMatMulWeightsOnlyQDQ
    else:
        if has_input_quant and has_output_quant:
            # Full dynamic quantization: weight + dynamic input + dynamic output
            return QMatMulWeightDynamicInputOutputQDQ
        elif has_input_quant:
            # Weight + dynamic input quantization
            return QMatMulWeightDynamicInputQDQ
        elif has_output_quant:
            # Weight + dynamic output quantization
            return QMatMulWeightDynamicOutputQDQ
        else:
            # Weights-only quantization
            return QMatMulWeightsOnlyQDQ

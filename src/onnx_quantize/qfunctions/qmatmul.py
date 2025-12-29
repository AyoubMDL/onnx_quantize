import onnx_ir as ir
from onnxscript import script

from onnx_quantize.opset import op
from onnx_quantize.qfunctions.register import QUANT_OPSET, register_qfunction


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulStatic8bits(X, W, x_scale, w_scale, x_zero_point, w_zero_point):
    """Static Quantized MatMul using ONNX ops."""
    # Quantize the inputs
    x_quantized = op.QuantizeLinear(X, x_scale, x_zero_point)

    # Int MatMul (W is already quantized)
    out_matmul = op.MatMulInteger(x_quantized, W, x_zero_point, w_zero_point)
    dequantized_matmul = op.DequantizeLinear(out_matmul, x_scale * w_scale)

    return dequantized_matmul


@register_qfunction(target_optype="MatMul")
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


@register_qfunction(target_optype="MatMul")
@script(opset=QUANT_OPSET)
def QMatMulWeightsOnly(X, W, w_scale, w_zero_point):
    """Weights only Quantized MatMul using ONNX ops."""
    # Dequantize weights
    dequantized_weights = op.DequantizeLinear(W, w_scale, w_zero_point)
    out_matmul = op.MatMul(X, dequantized_weights)

    return out_matmul


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

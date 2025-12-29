import onnx_ir as ir

from onnx_quantize.core._qconfig import GPTQConfig, QuantizationStrategy
from onnx_quantize.core._rtn import _quantize_bias
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qfunctions.qgemm import (
    _make_qgemm_weight_only_grouped,
    _make_qgemm_weight_only_grouped_4bits,
)
from onnx_quantize.qrules.matmul_to_qmatmul import MatMulToQMatMul


class GemmToQGemm(MatMulToQMatMul):
    """Rewrites MatMul nodes to QMatMul nodes."""

    def pattern(self, op, x, w):
        return op.Gemm(x, w, transB=0, _outputs=["out"])

    def check(self, context, w, out, **_):
        check_result = super().check(context, w)
        if not check_result:
            return check_result

        node = out.producer()
        transB = node.attributes.get("transB", ir.AttrInt64("transB", 0))
        if transB.value != 0:
            return check_result.fail("transB should be 0.")
        return check_result


class GemmBiasToQGemmBias(GemmToQGemm):
    """Rewrites MatMul nodes to QMatMul nodes."""

    def pattern(self, op, x, w, b):
        return op.Gemm(x, w, b, transB=0, _outputs=["out"])

    def check(self, context, w, b, out, **_):
        check_result = super().check(context, w, out)
        if not check_result:
            return check_result

        if ir.convenience.get_const_tensor(b) is None:
            return check_result.fail("Bias is not a constant tensor.")
        return check_result

    def _rewrite_static(self, op, x, w, b, out, qconfig):
        node = out.producer()

        # 1. get input scale and zero_point from calibrated model
        x_scale = op.initializer(ir.tensor(node.meta["input_scale"]), name=f"{x.name}/i_scale")
        x_zero_point = op.initializer(
            ir.tensor(node.meta["input_zero_point"]), name=f"{x.name}/i_zp"
        )

        # 2. Quantize the weights and bias
        w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)
        b_q, _, _ = _quantize_bias(
            b.const_value.numpy(), node.meta["input_scale"], w_scale.const_value.numpy()
        )
        b_q = op.initializer(ir.tensor(b_q), name=b.name)

        return op.QGemmStatic8bits(
            x,
            w_q,
            b_q,
            x_scale,
            w_scale,
            x_zero_point,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_dynamic(self, op, x, w, b, qconfig):
        # 1. Quantize the weights
        w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)

        return op.QGemmDynamic8bits(
            x,
            w_q,
            b,
            w_scale,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_weights_only(self, op, x, w, b, out, qconfig):
        node = out.producer()

        # 1. Quantize the weights
        if isinstance(qconfig.algorithm, GPTQConfig):
            w_q, w_scale, w_zero_point = self._quantize_gptq(op, x, w, node.meta["input"], qconfig)
        else:
            w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)

        if qconfig.strategy == QuantizationStrategy.GROUP:
            # This will register a new QFunction for this group size
            # Special case for grouped 4bits as ort doesn't support Reshape with 4bits inputs
            if qconfig.weights_dtype.bitwidth == 4:
                _make_qgemm_weight_only_grouped_4bits(qconfig.group_size)
            else:
                _make_qgemm_weight_only_grouped(qconfig.group_size)

            original_transposed_shape = op.initializer(
                ir.tensor(w.const_value.numpy().T.shape, dtype=ir.DataType.INT64),
                name=f"{w.name}/original_transposed_shape",
            )
            return op.QGemmWeightsOnlyGrouped(
                x,
                w_q,
                b,
                w_scale,
                w_zero_point,
                original_transposed_shape,
                _domain=QUANT_OPSET.domain,
                _version=QUANT_OPSET.version,
            )

        return op.QGemmWeightsOnly(
            x,
            w_q,
            b,
            w_scale,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def rewrite(self, op, x, w, b, out):
        return self._rewrite(op, x, w, b, out)


gemm_to_qgemm_rules = [GemmBiasToQGemmBias().rule(), GemmToQGemm().rule()]

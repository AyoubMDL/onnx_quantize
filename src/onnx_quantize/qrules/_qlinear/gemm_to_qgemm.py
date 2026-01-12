import onnx_ir as ir

from onnx_quantize.core._qconfig import QConfig, QFormat
from onnx_quantize.core._rtn import _quantize_bias as _quantize_bias_rtn
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qrules._common import quantize_weights
from onnx_quantize.qrules._qlinear.matmul_to_qmatmul import MatMulToQLinearMatMul


class GemmToQLinearGemm(MatMulToQLinearMatMul):
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


class GemmBiasToQLinearGemmBias(GemmToQLinearGemm):
    """Rewrites MatMul nodes to QMatMul nodes."""

    @property
    def op_type(self):
        return "Gemm"

    def pattern(self, op, x, w, b):
        # TODO: write explicitly other attrs
        return op.Gemm(x, w, b, transB=0, _outputs=["out"])

    def check(self, context, w, b, out, **_):
        check_result = super().check(context, w, out)
        if not check_result:
            return check_result

        if ir.convenience.get_const_tensor(b) is None:
            return check_result.fail("Bias is not a constant tensor.")
        return check_result

    def _quantize_bias(self, op, b, input_scale, w_scale):
        b_q, _, _ = _quantize_bias_rtn(
            b.const_value.numpy(),
            input_scale.const_value.numpy(),
            w_scale.const_value.numpy(),
        )

        b_q = op.initializer(ir.tensor(b_q), name=b.name)
        return b_q

    def _rewrite_static(self, op, x, w, b, out, qconfig: QConfig):
        assert qconfig.format == QFormat.QLINEAR
        node = out.producer()

        # 1. Quantize the weights
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig, out)

        # 2: Get activation quantization parameters
        input_scale, input_zero_point = self._get_activation_qparams(
            op, node, "input", qconfig.input_activations
        )
        out_scale, out_zero_point = self._get_activation_qparams(
            op, node, "output", qconfig.output_activations
        )

        # 3. Quantize bias using input and weight scales (zero point should be zero)
        b_q = self._quantize_bias(op, b, input_scale, w_scale)

        qfunc_name = self.qfunction(self.op_type, qconfig).__name__
        return getattr(op, qfunc_name)(
            x,
            w_q,
            b_q,
            w_scale,
            w_zero_point,
            input_scale,
            input_zero_point,
            out_scale,
            out_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def rewrite(self, op, x, w, b, out):
        return self._rewrite(op, x, w, b, out)


gemm_to_qlinear_gemm_rules = [GemmBiasToQLinearGemmBias().rule(), GemmToQLinearGemm().rule()]

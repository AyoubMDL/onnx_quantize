import onnx_ir as ir
import onnxscript


class StandarizeGemm(onnxscript.rewriter.RewriteRuleClassBase):
    """Makes transB attributes always 0."""

    def pattern(self, op, x, w):
        return op.Gemm(x, w, _outputs=["out"])

    def rewrite(self, op, x, w, **_):
        if self.transB:
            w = op.initializer(ir.tensor(w.const_value.numpy().T), name=w.name)
        return op.Gemm(x, w, transB=0)

    def check(self, context, w, out, **_):
        del context  # Not used
        check_result = onnxscript.rewriter.MatchResult()
        if ir.convenience.get_const_tensor(w) is None:
            return check_result.fail("Weight is not a constant tensor.")

        node = out.producer()

        transB = node.attributes.get("transB")

        self.transB = 1
        if transB is None:
            self.transB = 0
            return check_result

        if transB.value == 0:
            return check_result.fail("transB attribute is already 0.")

        return check_result


class StandarizeGemmBias(StandarizeGemm):
    """Makes transB attributes always 0."""

    def pattern(self, op, x, w, b):
        return op.Gemm(x, w, b, _outputs=["out"])

    def rewrite(self, op, x, w, b, **_):
        if self.transB:
            w = op.initializer(ir.tensor(w.const_value.numpy().T), name=w.name)
        return op.Gemm(x, w, b, transB=0)


standarize_gemm_rules = [StandarizeGemmBias().rule(), StandarizeGemm().rule()]

import numpy as np
import onnx_ir as ir
import onnxscript

from onnx_quantize.core import QConfig, quantize_tensor
from onnx_quantize.qfunctions import QUANT_OPSET


class MatMulToQMatMul(onnxscript.rewriter.RewriteRuleClassBase):
    """Rewrites MatMul nodes to QMatMul nodes."""

    def pattern(self, op, x, w):
        return op.MatMul(x, w, _outputs=["out"])

    def check(self, context, w, **_):
        del context  # Not used
        check_result = onnxscript.rewriter.MatchResult()

        if ir.convenience.get_const_tensor(w) is None:
            return check_result.fail("Weight is not a constant tensor.")
        return check_result

    def rewrite(self, op, x, w, out):
        node = out.producer()
        qconfig = QConfig(**node.meta["qconfig"])
        if qconfig.is_static:
            return self._rewrite_static(op, x, w, out)
        return self._rewrite_dynamic(op, x, w, out)

    def _quantize_weights(self, op, x, w, qconfig):
        w_q, w_scale, w_zero_point = quantize_tensor(
            w.const_value.numpy(),
            qconfig.weights_dtype,
            qconfig.weights_symmetric,
            qconfig.weights_per_channel,
        )

        w_q = op.initializer(ir.tensor(w_q), name=w.name)
        w_scale = op.initializer(ir.tensor(np.squeeze(w_scale)), name=f"{x.name}/w_scale")
        w_zero_point = op.initializer(
            ir.tensor(np.squeeze(w_zero_point)), name=f"{x.name}/w_zero_point"
        )

        return w_q, w_scale, w_zero_point

    def _rewrite_static(self, op, x, w, out):
        node = out.producer()

        # 1. get input scale and zero_point from calibrated model
        x_scale = op.initializer(ir.tensor(node.meta["input_scale"]), name=f"{x.name}/i_scale")
        x_zero_point = op.initializer(
            ir.tensor(node.meta["input_zero_point"]), name=f"{x.name}/i_zp"
        )

        # 2. Quantize the weights
        qconfig = QConfig(**node.meta["qconfig"])
        w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)

        return op.QMatMulStatic8bits(
            x,
            w_q,
            x_scale,
            w_scale,
            x_zero_point,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_dynamic(self, op, x, w, out):
        node = out.producer()

        # 2. Quantize the weights
        qconfig = QConfig(**node.meta["qconfig"])
        w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)

        return op.QMatMulDynamic8bits(
            x,
            w_q,
            w_scale,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )


matmul_to_qmatmul_rules = [MatMulToQMatMul().rule()]

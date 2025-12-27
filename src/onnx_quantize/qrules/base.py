import onnxscript

from onnx_quantize.core._qconfig import QConfig


class QRewriter(onnxscript.rewriter.RewriteRuleClassBase):
    """Base class for quantization rewriters."""

    def _rewrite_weights_only(self, op, *args, qconfig):
        raise NotImplementedError()

    def _rewrite_static(self, op, *args, qconfig):
        raise NotImplementedError()

    def _rewrite_dynamic(self, op, *args, qconfig):
        raise NotImplementedError()

    def _rewrite(self, op, *args):
        # args can be (x, w, out) or (x, w, b, out) depending on the pattern
        # (when bias is present or not)
        out = args[-1]  # out is always the last argument
        node = out.producer()
        qconfig = QConfig(**node.meta["qconfig"])

        if qconfig.weights_only:
            return self._rewrite_weights_only(op, *args, qconfig=qconfig)

        elif qconfig.is_static:
            return self._rewrite_static(op, *args, qconfig=qconfig)

        return self._rewrite_dynamic(op, *args[:-1], qconfig=qconfig)

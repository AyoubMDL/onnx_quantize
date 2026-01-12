import onnxscript

from onnx_quantize.core._qconfig import QConfig
from onnx_quantize.qfunctions.factory import get_qfunction


class QRewriter(onnxscript.rewriter.RewriteRuleClassBase):
    """Base class for quantization rewriters."""

    def qfunction(self, op_type: str, qconfig: QConfig):
        return get_qfunction(op_type, qconfig)

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

        weights_only = qconfig.input_activations is None and qconfig.output_activations is None
        static_input = qconfig.input_activations is not None and qconfig.input_activations.is_static
        static_output = (
            qconfig.output_activations is not None and qconfig.output_activations.is_static
        )
        is_static = static_input or static_output

        if weights_only:
            return self._rewrite_weights_only(op, *args, qconfig=qconfig)

        elif is_static:
            return self._rewrite_static(op, *args, qconfig=qconfig)

        return self._rewrite_dynamic(op, *args, qconfig=qconfig)

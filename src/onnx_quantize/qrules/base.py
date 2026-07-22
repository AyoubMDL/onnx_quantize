import onnx_ir as ir
import onnxscript

from onnx_quantize.core._qconfig import QConfig
from onnx_quantize.qfunctions.factory import get_qfunction
from onnx_quantize.qrules._common import _resolve_group_size


class QRewriter(onnxscript.rewriter.RewriteRuleClassBase):
    """Base class for quantization rewriters."""

    def qfunction(self, op_type: str, qconfig: QConfig):
        return get_qfunction(op_type, qconfig)

    def _get_activation_qparams(self, op, node, kind, qconfig_act):
        """Return ``(scale, zero_point)`` initializers for a static activation.

        Returns ``(None, None)`` when the activation is not statically quantized.

        Args:
            op: The rewriter tape used to create the initializers.
            node: The original node being rewritten; its unique output name keys the
                qparam initializers.
            kind: ``"input"`` or ``"output"`` -- selects the calibrated qparams stored
                in ``node.meta`` under ``f"{kind}_scale"`` / ``f"{kind}_zero_point"``.
            qconfig_act: Activation quantization config, or ``None`` when absent.
        """
        if qconfig_act is None or not qconfig_act.is_static:
            return None, None

        # Name qparams after this node's (unique) output so a tensor that fans out to
        # several quantized nodes yields distinct initializers instead of colliding.
        prefix = node.outputs[0].name
        scale = op.initializer(
            ir.tensor(node.meta[f"{kind}_scale"]), name=f"{prefix}/{kind}/scale"
        )
        zero_point = op.initializer(
            ir.tensor(node.meta[f"{kind}_zero_point"]), name=f"{prefix}/{kind}/zero_point"
        )
        return scale, zero_point

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

        # Resolve weight group size
        qconfig.weights.group_size = _resolve_group_size(args[1], qconfig.weights.group_size)

        if weights_only:
            return self._rewrite_weights_only(op, *args, qconfig=qconfig)

        elif is_static:
            return self._rewrite_static(op, *args, qconfig=qconfig)

        return self._rewrite_dynamic(op, *args, qconfig=qconfig)

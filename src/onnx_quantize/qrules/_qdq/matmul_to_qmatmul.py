"""MatMul to QMatMul rewriter using QDQ pattern."""

import onnx_ir as ir
import onnxscript

from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qrules._common import quantize_weights
from onnx_quantize.qrules.base import QRewriter


class MatMulToQMatMul(QRewriter):
    """Rewrites MatMul nodes to quantized QMatMul nodes using QDQ pattern.

    This rewriter handles three quantization scenarios:
    1. Weights-only: Only weights are quantized (most common for LLMs)
    2. Static: Weights + activations quantized with calibrated scales (GPTQ or RTN)
    3. Dynamic: Weights quantized, activations dynamically quantized at runtime
    """

    @property
    def op_type(self):
        return "MatMul"

    def pattern(self, op, x, w):
        return op.MatMul(x, w, _outputs=["out"])

    def check(self, context, w, **_):
        del context
        check_result = onnxscript.rewriter.MatchResult()

        if ir.convenience.get_const_tensor(w) is None:
            return check_result.fail("Weight is not a constant tensor.")
        return check_result

    def _get_activation_qparams(self, op, node, prefix, qconfig_act):
        if qconfig_act is None or not qconfig_act.is_static:
            return None, None

        # Extract calibrated scale and zero_point from node metadata
        scale_key = f"{prefix}_scale"
        zp_key = f"{prefix}_zero_point"

        scale = op.initializer(ir.tensor(node.meta[scale_key]), name=f"{prefix}/scale")
        zero_point = op.initializer(ir.tensor(node.meta[zp_key]), name=f"{prefix}/zero_point")

        return scale, zero_point

    def _rewrite_weights_only(self, op, x, w, out, qconfig: QConfig):
        qfunc_name = self.qfunction(self.op_type, qconfig).__name__
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig, out)

        # Handle grouped quantization
        func_args = [x, w_q, w_scale, w_zero_point]
        if qconfig.weights.strategy == QuantizationStrategy.GROUP:
            original_transposed_shape = op.initializer(
                ir.tensor(w.const_value.numpy().T.shape, dtype=ir.DataType.INT64),
                name=f"{w.name}/original_transposed_shape",
            )
            func_args.append(original_transposed_shape)

        return getattr(op, qfunc_name)(
            *func_args,
            num_bits=qconfig.weights.dtype.bitwidth,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_dynamic(self, op, x, w, out, qconfig: QConfig):
        # 1. Quantize the weights
        qfunc_name = self.qfunction(self.op_type, qconfig).__name__
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig)

        return getattr(op, qfunc_name)(
            x,
            w_q,
            w_scale,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_static(self, op, x, w, out, qconfig: QConfig):
        node = out.producer()

        # 1. Quantize the weights
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig)

        # 2: Get activation quantization parameters
        input_scale, input_zero_point = self._get_activation_qparams(
            op, node, "input", qconfig.input_activations
        )
        out_scale, out_zero_point = self._get_activation_qparams(
            op, node, "output", qconfig.output_activations
        )

        # 3: Build argument list based on what's needed
        func_args = [x, w_q, w_scale, w_zero_point]
        if input_scale is not None and input_zero_point is not None:
            func_args.extend([input_scale, input_zero_point])

        if out_scale is not None and out_zero_point is not None:
            func_args.extend([out_scale, out_zero_point])

        qfunc_name = self.qfunction(self.op_type, qconfig).__name__
        return getattr(op, qfunc_name)(
            *func_args,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def rewrite(self, op, x, w, out):
        return self._rewrite(op, x, w, out)


matmul_to_qdq_matmul_rules = [MatMulToQMatMul().rule()]

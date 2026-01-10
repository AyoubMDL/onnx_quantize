"""MatMul to QMatMul rewriter using QDQ pattern."""

import numpy as np
import onnx_ir as ir
import onnxscript

from onnx_quantize.core._gptq import _gptq_quantize
from onnx_quantize.core._qconfig import GPTQConfig, QConfig, QuantizationStrategy
from onnx_quantize.core._rtn import _quantize_array
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qfunctions._qdq.qmatmul import qmatmul_qdq_factory
from onnx_quantize.qrules.base import QRewriter


class MatMulToQMatMul(QRewriter):
    """Rewrites MatMul nodes to quantized QMatMul nodes using QDQ pattern.

    This rewriter handles three quantization scenarios:
    1. Weights-only: Only weights are quantized (most common for LLMs)
    2. Static: Weights + activations quantized with calibrated scales (GPTQ or RTN)
    3. Dynamic: Weights quantized, activations dynamically quantized at runtime
    """

    def pattern(self, op, x, w):
        return op.MatMul(x, w, _outputs=["out"])

    def check(self, context, w, **_):
        del context
        check_result = onnxscript.rewriter.MatchResult()

        if ir.convenience.get_const_tensor(w) is None:
            return check_result.fail("Weight is not a constant tensor.")
        return check_result

    def _quantize_weights_gptq(self, op, w, inputs, qconfig: QConfig):
        w_q, w_scale, w_zero_point = _gptq_quantize(
            w.const_value.numpy(),
            inputs,
            quant_type=qconfig.weights.dtype,
            strategy=qconfig.weights.strategy,
            is_symmetric=qconfig.weights.symmetric,
            reduce_range=qconfig.weights.reduce_range,
            clip_ratio=qconfig.weights.clip_ratio,
            block_size=qconfig.weights.algorithm.block_size,
            percdamp=qconfig.weights.algorithm.percdamp,
            group_size=qconfig.weights.algorithm.group_size,
            actorder=qconfig.weights.algorithm.actorder,
            mse=qconfig.weights.mse,
        )

        # Create ONNX tensors from quantized weights
        w_q = op.initializer(ir.tensor(w_q), name=w.name)
        w_scale = op.initializer(
            ir.tensor(np.squeeze(w_scale.astype(qconfig.weights.scale_dtype))),
            name=f"{w.name}/scale",
        )
        w_zero_point = op.initializer(
            ir.tensor(np.squeeze(w_zero_point)), name=f"{w.name}/zero_point"
        )

        return w_q, w_scale, w_zero_point

    def _quantize_weights_rtn(self, op, w, qconfig: QConfig):
        w_q, w_scale, w_zero_point = _quantize_array(
            w.const_value.numpy(),
            qconfig.weights.dtype,
            strategy=qconfig.weights.strategy,
            group_size=qconfig.weights.group_size,
            is_symmetric=qconfig.weights.symmetric,
            reduce_range=qconfig.weights.reduce_range,
            clip_ratio=qconfig.weights.clip_ratio,
            mse=qconfig.weights.mse,
        )

        # Create ONNX tensors from quantized weights
        w_q = op.initializer(ir.tensor(w_q), name=w.name)
        w_scale = op.initializer(
            ir.tensor(w_scale.astype(qconfig.weights.scale_dtype)), name=f"{w.name}/scale"
        )
        w_zero_point = op.initializer(ir.tensor(w_zero_point), name=f"{w.name}/zero_point")

        return w_q, w_scale, w_zero_point

    def _quantize_weights(self, op, w, out, qconfig: QConfig):
        if isinstance(qconfig.weights.algorithm, GPTQConfig):
            node = out.producer()
            assert "input" in node.meta, "GPTQ requires calibration data in node meta."
            return self._quantize_weights_gptq(op, w, node.meta["input"], qconfig)
        else:
            return self._quantize_weights_rtn(op, w, qconfig)

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
        qfunc_name = qmatmul_qdq_factory(qconfig).__name__
        w_q, w_scale, w_zero_point = self._quantize_weights(op, w, out, qconfig)

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
        w_q, w_scale, w_zero_point = self._quantize_weights(op, w, out, qconfig)
        qfunc_name = qmatmul_qdq_factory(qconfig).__name__

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
        w_q, w_scale, w_zero_point = self._quantize_weights(op, w, out, qconfig)

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

        qfunc_name = qmatmul_qdq_factory(qconfig).__name__
        return getattr(op, qfunc_name)(
            *func_args,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def rewrite(self, op, x, w, out):
        return self._rewrite(op, x, w, out)


matmul_to_qmatmul_rules = [MatMulToQMatMul().rule()]

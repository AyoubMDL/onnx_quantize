import onnx_ir as ir

from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy, QWeightArgs
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qrules._common import quantize_weights
from onnx_quantize.qrules._qdq.matmul_to_qmatmul import MatMulToQMatMul


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

    def _quantize_bias(self, op, b, qconfig: QConfig):
        # Construct a new QConfig for bias quantization with tensor strategy
        # and rtn quantization
        qconfig = QConfig(
            weights=QWeightArgs(
                dtype=qconfig.weights.dtype,
                is_symmetric=qconfig.weights.symmetric,
                strategy=QuantizationStrategy.TENSOR,
                scale_type=qconfig.weights.scale_dtype,
                clip_ratio=qconfig.weights.clip_ratio,
                mse=qconfig.weights.mse,
                reduce_range=qconfig.weights.reduce_range,
            )
        )
        b_q, b_scale, b_zero_point = quantize_weights(op, b, qconfig)
        return b_q, b_scale, b_zero_point

    def _rewrite_static(self, op, x, w, b, out, qconfig: QConfig):
        node = out.producer()

        # 1. Quantize the weights
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig, out)

        # Quantize bias
        b_q, b_scale, b_zero_point = self._quantize_bias(op, b, qconfig)

        # 2: Get activation quantization parameters
        input_scale, input_zero_point = self._get_activation_qparams(
            op, node, "input", qconfig.input_activations
        )
        out_scale, out_zero_point = self._get_activation_qparams(
            op, node, "output", qconfig.output_activations
        )

        # 3: Build argument list based on what's needed
        func_args = [x, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point]
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

    def _rewrite_dynamic(self, op, x, w, b, out, qconfig: QConfig):
        # Quantize the weights
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig, out)

        # Quantize bias
        b_q, b_scale, b_zero_point = self._quantize_bias(op, b, qconfig)

        qfunc_name = self.qfunction(self.op_type, qconfig).__name__

        return getattr(op, qfunc_name)(
            x,
            w_q,
            b_q,
            w_scale,
            w_zero_point,
            b_scale,
            b_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_weights_only(self, op, x, w, b, out, qconfig: QConfig):
        qfunc_name = self.qfunction(self.op_type, qconfig).__name__

        # Quantize the weights
        w_q, w_scale, w_zero_point = quantize_weights(op, w, qconfig, out)

        # Quantize bias
        b_q, b_scale, b_zero_point = self._quantize_bias(op, b, qconfig)

        # Handle grouped quantization
        func_args = [x, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point]
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

    def rewrite(self, op, x, w, b, out):
        return self._rewrite(op, x, w, b, out)


gemm_to_qdq_gemm_rules = [GemmBiasToQGemmBias().rule(), GemmToQGemm().rule()]

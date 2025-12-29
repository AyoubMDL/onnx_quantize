import numpy as np
import onnx_ir as ir
import onnxscript

from onnx_quantize.core._gptq import _gptq_quantize
from onnx_quantize.core._qconfig import GPTQConfig, QuantizationStrategy
from onnx_quantize.core._rtn import _quantize_array
from onnx_quantize.qfunctions import QUANT_OPSET
from onnx_quantize.qfunctions.qmatmul import (
    _make_qmatmul_weight_only_grouped,
    _make_qmatmul_weight_only_grouped_4bits,
)
from onnx_quantize.qrules.base import QRewriter


class MatMulToQMatMul(QRewriter):
    """Rewrites MatMul nodes to QMatMul nodes."""

    def pattern(self, op, x, w):
        return op.MatMul(x, w, _outputs=["out"])

    def check(self, context, w, **_):
        del context  # Not used
        check_result = onnxscript.rewriter.MatchResult()

        if ir.convenience.get_const_tensor(w) is None:
            return check_result.fail("Weight is not a constant tensor.")
        return check_result

    def _quantize_gptq(self, op, x, w, inputs, qconfig):
        w_q, w_scale, w_zero_point = _gptq_quantize(
            w.const_value.numpy(),
            inputs,
            quant_type=qconfig.weights_dtype,
            strategy=qconfig.strategy,
            group_size=qconfig.algorithm.group_size,
            is_symmetric=qconfig.weights_symmetric,
            reduce_range=qconfig.reduce_range,
            clip_ratio=qconfig.clip_ratio,
            block_size=qconfig.algorithm.block_size,
            percdamp=qconfig.algorithm.percdamp,
            actorder=qconfig.algorithm.actorder,
            mse=qconfig.mse,
        )

        w_q = op.initializer(ir.tensor(w_q), name=w.name)
        w_scale = op.initializer(ir.tensor(np.squeeze(w_scale)), name=f"{x.name}/w_scale")
        w_zero_point = op.initializer(
            ir.tensor(np.squeeze(w_zero_point)), name=f"{x.name}/w_zero_point"
        )

        return w_q, w_scale, w_zero_point

    def _quantize_weights(self, op, x, w, qconfig):
        w_q, w_scale, w_zero_point = _quantize_array(
            w.const_value.numpy(),
            qconfig.weights_dtype,
            strategy=qconfig.strategy,
            group_size=qconfig.group_size,
            is_symmetric=qconfig.weights_symmetric,
            reduce_range=qconfig.reduce_range,
            clip_ratio=qconfig.clip_ratio,
            mse=qconfig.mse,
        )

        w_q = op.initializer(ir.tensor(w_q), name=w.name)
        w_scale = op.initializer(ir.tensor(w_scale), name=f"{x.name}/w_scale")
        w_zero_point = op.initializer(ir.tensor(w_zero_point), name=f"{x.name}/w_zero_point")

        return w_q, w_scale, w_zero_point

    def _rewrite_static(self, op, x, w, out, qconfig):
        node = out.producer()

        # 1. get input scale and zero_point from calibrated model
        x_scale = op.initializer(ir.tensor(node.meta["input_scale"]), name=f"{x.name}/i_scale")
        x_zero_point = op.initializer(
            ir.tensor(node.meta["input_zero_point"]), name=f"{x.name}/i_zp"
        )

        # 2. Quantize the weights
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

    def _rewrite_dynamic(self, op, x, w, qconfig):
        # 1. Quantize the weights
        w_q, w_scale, w_zero_point = self._quantize_weights(op, x, w, qconfig)

        return op.QMatMulDynamic8bits(
            x,
            w_q,
            w_scale,
            w_zero_point,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def _rewrite_weights_only(self, op, x, w, out, qconfig):
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
                _make_qmatmul_weight_only_grouped_4bits(qconfig.group_size)
            else:
                _make_qmatmul_weight_only_grouped(qconfig.group_size)
            original_transposed_shape = op.initializer(
                ir.tensor(w.const_value.numpy().T.shape, dtype=ir.DataType.INT64),
                name=f"{w.name}/original_transposed_shape",
            )
            return op.QMatMulWeightsOnlyGrouped(
                x,
                w_q,
                w_scale,
                w_zero_point,
                original_transposed_shape,
                num_bits=qconfig.weights_dtype.bitwidth,
                _domain=QUANT_OPSET.domain,
                _version=QUANT_OPSET.version,
            )

        return op.QMatMulWeightsOnly(
            x,
            w_q,
            w_scale,
            w_zero_point,
            num_bits=qconfig.weights_dtype.bitwidth,
            _domain=QUANT_OPSET.domain,
            _version=QUANT_OPSET.version,
        )

    def rewrite(self, op, x, w, out):
        return self._rewrite(op, x, w, out)


matmul_to_qmatmul_rules = [MatMulToQMatMul().rule()]

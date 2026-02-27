import logging

import numpy as np
import onnx_ir as ir

from onnx_quantize.core._algorithms.rtn import _rtn_quantize
from onnx_quantize.core._algorithms.utils import _dequantize_array
from onnx_quantize.core._qconfig import QConfig, QuantizationStrategy


logger = logging.getLogger(__name__)
_SUPPORTED_OPS = {"MatMul", "Gemm"}


# TODO: add folding of mul nodes
class AwqPass(ir.passes.InPlacePass):
    """Pass for AWQ quantization."""

    def __init__(self, clip_search, target_op_types):
        self.clip_search = clip_search
        self.target_op_types = target_op_types

    def _compute_activation_scale(self, inputs: np.ndarray) -> np.ndarray:
        hidden_dim = inputs.shape[-1]
        act_scale = np.mean(np.reshape(np.abs(inputs), (-1, hidden_dim)), axis=0)
        return act_scale

    def _compute_weight_scale(
        self, weights: np.ndarray, strategy: QuantizationStrategy, group_size: int
    ):
        org_shape = weights.shape

        keep_dims = True
        axis = 1
        if strategy == QuantizationStrategy.TENSOR:
            axis = None
            keep_dims = False

        if strategy == QuantizationStrategy.GROUP:
            weights = np.reshape(weights, (-1, group_size))

        scale = np.abs(weights) / np.max(np.abs(weights), axis=axis, keepdims=keep_dims)
        scale = np.reshape(scale, org_shape)
        scale = np.mean(scale, axis=0)

        return scale

    def _insert_mul_node_before(
        self, node: ir.Node, model: ir.Model, scale_initializer: ir.Value
    ) -> None:
        mul_node = ir.node(
            "Mul",
            inputs=[
                node.inputs[0],
                scale_initializer,
            ],
        )

        model.graph.register_initializer(scale_initializer)
        model.graph.insert_before(node, mul_node)
        node.replace_input_with(0, mul_node.outputs[0])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the AWQ quantization pass."""
        modified = False

        for node in model.graph:
            modified |= self._apply_awq(node, model)

            if self.clip_search:
                modified |= self._apply_awq_clip(node)

        if modified:
            logger.info("AWQ pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def is_valid_node(self, node: ir.Node) -> bool:
        constrains = [
            lambda: node.op_type not in self.target_op_types,
            lambda: node.op_type not in _SUPPORTED_OPS,
            lambda: node.domain != "",
            lambda: ir.convenience.get_const_tensor(node.inputs[1]) is None,
            lambda: node.attributes.get("transB", ir.AttrInt64("transB", 0)).as_int() != 0,
        ]

        return not any(check() for check in constrains)

    def _apply_awq(self, node: ir.Node, model: ir.Model) -> bool:
        if not self.is_valid_node(node):
            return False

        qconfig = QConfig(**node.meta["qconfig"])

        # 1. Compute activation scale
        act_scale = self._compute_activation_scale(node.meta["input"])

        # 2. Compute weight scale
        original_weights = ir.convenience.get_const_tensor(node.inputs[1]).numpy()
        weights_scale = self._compute_weight_scale(
            original_weights.T, qconfig.weights.strategy, qconfig.weights.group_size
        )

        # TODO: Refactor for other ops
        original_outputs = np.matmul(node.meta["input"], original_weights)

        best_error = np.inf
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid

            # 3. Compute AWQ scale:
            scale = np.clip(
                np.power(act_scale, ratio) / np.power(weights_scale, (1 - ratio)), 1e-4, None
            )
            scale = scale / np.sqrt(np.max(scale) * np.min(scale))

            weights = original_weights * scale.reshape(-1, 1)

            # 4. Fake quantize weights
            qweights, qscale, qzp = _rtn_quantize(
                weights,
                quant_type=qconfig.weights.dtype,
                strategy=qconfig.weights.strategy,
                group_size=qconfig.weights.group_size,
                is_symmetric=qconfig.weights.symmetric,
                reduce_range=qconfig.weights.reduce_range,
                clip_ratio=1.0,
                mse=False,
                scale_dtype=qconfig.weights.scale_dtype,
                zp_dtype=qconfig.weights.zp_dtype,
            )
            qweights = _dequantize_array(
                qweights,
                qscale,
                qzp,
                preprocess=True,
                strategy=qconfig.weights.strategy,
                group_size=qconfig.weights.group_size,
            )
            q_weight = qweights / scale.reshape(-1, 1)
            out = np.matmul(node.meta["input"], q_weight)
            loss = np.mean(np.power((original_outputs - out), 2))

            if loss < best_error:
                best_error = loss
                best_scale = scale

        # 4. Fuse scale into weights
        updated_weights = original_weights * best_scale.reshape(-1, 1)

        # 5. Create a Mul node to scale the input activations
        scale_initializer = ir.val(
            f"{node.outputs[0].name}_scale", const_value=ir.tensor(1.0 / best_scale)
        )

        # 6. Input activation metadata need to be updated as the activations were scaled
        node.meta["input"] /= best_scale.reshape((1, -1))

        # 7. Insert the Mul node before the current node
        self._insert_mul_node_before(node, model, scale_initializer)

        # 8. Update weight initializer
        weights_initializer = ir.val(node.inputs[1].name, const_value=ir.tensor(updated_weights))
        ir.convenience.replace_all_uses_with(node.inputs[1], weights_initializer)
        model.graph.initializers[node.inputs[1].name] = weights_initializer

        return True

    def _apply_awq_clip(self, node: ir.Node) -> bool:
        if not self.is_valid_node(node):
            return False

        qconfig = QConfig(**node.meta["qconfig"])

        inputs = node.meta["input"]
        weights = ir.convenience.get_const_tensor(node.inputs[1]).numpy()
        original_outputs = np.matmul(inputs, weights)

        best_error = np.inf
        best_ratio = 1

        for i_s in range(10):
            ratio = 1 - i_s / 100

            qweights, qscale, qzp = _rtn_quantize(
                weights,
                quant_type=qconfig.weights.dtype,
                strategy=qconfig.weights.strategy,
                group_size=qconfig.weights.group_size,
                is_symmetric=qconfig.weights.symmetric,
                reduce_range=qconfig.weights.reduce_range,
                clip_ratio=ratio,
                mse=False,
                scale_dtype=qconfig.weights.scale_dtype,
                zp_dtype=qconfig.weights.zp_dtype,
            )
            qweights = _dequantize_array(
                qweights,
                qscale,
                qzp,
                preprocess=True,
                strategy=qconfig.weights.strategy,
                group_size=qconfig.weights.group_size,
            )

            current_outputs = np.matmul(inputs, qweights)

            loss = np.mean(np.power((original_outputs - current_outputs), 2))

            if loss < best_error:
                best_error = loss
                best_ratio = ratio

        qconfig.weights.clip_ratio = best_ratio
        node.meta["qconfig"] = qconfig.model_dump()

        return True

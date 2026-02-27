import logging

import numpy as np
import onnx_ir as ir

from onnx_quantize.core._qconfig import QConfig


logger = logging.getLogger(__name__)


# TODO: add folding of mul nodes
# see https://github.com/intel/neural-compressor/blob/master/neural_compressor/adaptor/ox_utils/smooth_quant.py
class SmoothQuantPass(ir.passes.InPlacePass):
    """Pass for smooth quantization.

    Args:
        alpha (float): Smoothing factor between 0 and 1.
    """

    def __init__(self, alpha, target_op_types):
        self.alpha = alpha
        self.target_op_types = target_op_types

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the smooth quantization pass."""
        modified = False

        for node in model.graph:
            if self._smooth_quant_node(node, model):
                modified = True

        if modified:
            logger.info("SmoothQuant pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def _compute_activation_scale(self, inputs: np.ndarray) -> np.ndarray:
        hidden_dim = inputs.shape[-1]
        tensor = np.abs(inputs.reshape(-1, hidden_dim))
        act_scale = np.max(tensor, axis=0)
        # Clamp to avoid scale=0 (zero-activation channels need no smoothing; scale=1 is correct)
        act_scale = np.maximum(act_scale, 1e-5)

        return act_scale

    def _compute_weight_scale(self, weights: ir.Value) -> tuple[np.ndarray, np.ndarray]:
        weights = ir.convenience.get_const_tensor(weights).numpy()
        weights_scale = np.max(np.abs(weights), axis=1)
        return weights, weights_scale

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

    def _smooth_quant_node(self, node: ir.Node, model: ir.Model) -> bool:
        if node.op_type not in self.target_op_types or node.domain != "":
            return False

        if ir.convenience.get_const_tensor(node.inputs[1]) is None:
            return False

        qconfig = QConfig(**node.meta["qconfig"])

        if not qconfig.preprocessors:
            return False

        # 1. Compute activation scale
        act_scale = self._compute_activation_scale(node.meta["input"])

        # 2. Compute weight scale
        weights, weights_scale = self._compute_weight_scale(node.inputs[1])

        # 3. Compute smooth scale using the formula:
        scale = np.power(act_scale, self.alpha) / np.power(weights_scale + 1e-9, (1 - self.alpha))

        # 4. Fuse scale into weights
        updated_weights = np.multiply(scale.reshape(-1, 1), weights)

        # 5. Create a Mul node to scale the input activations
        scale_initializer = ir.val(
            f"{node.outputs[0].name}_scale", const_value=ir.tensor(1.0 / scale)
        )

        # 6. Input activation metadata need to be updated as the activations were scaled
        node.meta["input"] /= scale.reshape((1, -1))

        # 7. Insert the Mul node before the current node
        self._insert_mul_node_before(node, model, scale_initializer)

        # 8. Update weight initializer
        weights_initializer = ir.val(node.inputs[1].name, const_value=ir.tensor(updated_weights))
        ir.convenience.replace_all_uses_with(node.inputs[1], weights_initializer)
        model.graph.initializers[node.inputs[1].name] = weights_initializer

        return True

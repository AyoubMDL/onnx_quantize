import logging

import onnx_ir as ir


logger = logging.getLogger(__name__)


def _should_skip_initializer(initializer: ir.Value) -> bool:
    """Check if the initializer should be skipped for deduplication."""
    if initializer.is_graph_input() or initializer.is_graph_output():
        # Skip graph inputs and outputs
        logger.debug(
            "Skipped deduplication of initializer '%s' as it is a graph input or output",
            initializer.name,
        )
        return True

    const_val = initializer.const_value
    if const_val is None:
        # Skip if initializer has no constant value
        logger.debug(
            "Skipped deduplication of initializer '%s' as it has no constant value. "
            "The model may contain invalid initializers",
            initializer.name,
        )
        return True

    if len(initializer.uses()) <= 1:
        return True

    return False


class DuplicateInitializersPass(ir.passes.InPlacePass):
    """Duplicate initializers that are consumed by more than one node.

    Some models share a single initializer across multiple consumers
    (tied weights). A common example is an LM head that reuses the
    embedding matrix: both the embedding lookup and the output
    projection point to the same tensor.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for graph in model.graphs():
            for initializer in tuple(graph.initializers.values()):
                if _should_skip_initializer(initializer):
                    continue

                # keep the first use on the original, duplicate the rest
                for i, (node, input_idx) in enumerate(initializer.uses()[1:], start=1):
                    # This risky for name collision
                    # See https://github.com/onnx/ir-py/issues/246
                    new_name = f"{initializer.name}_{i}"
                    new_value = ir.val(
                        new_name,
                        type=initializer.type,
                        shape=initializer.shape,
                        const_value=initializer.const_value,
                    )
                    node.replace_input_with(input_idx, new_value)
                    graph.register_initializer(new_value)
                    modified = True

        return ir.passes.PassResult(model=model, modified=modified)

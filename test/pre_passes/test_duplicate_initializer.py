import numpy as np
import onnx
import onnx_ir as ir

from onnx_quantize.pre_passes import DuplicateInitializersPass
from test.helpers import onnx_forward_on_models


def _model_with_shared_weight(rng, num_consumers):
    """Build a model with ``num_consumers`` MatMul nodes that all share weight W."""
    inputs = ", ".join(f"float[N, 4] X{i}" for i in range(num_consumers))
    outputs = ", ".join(f"float[N, 8] Y{i}" for i in range(num_consumers))
    body = "\n".join(f"    Y{i} = MatMul(X{i}, W)" for i in range(num_consumers))

    model = onnx.parser.parse_model(f"""
        < ir_version: 10, opset_import: ["" : 21] >
        test_model ({inputs}) => ({outputs})
        <float[4, 8] W>
        {{
            {body}
        }}
    """)
    weight = rng.normal(size=(4, 8)).astype(np.float32)
    model.graph.initializer.extend([onnx.numpy_helper.from_array(weight, name="W")])
    onnx.checker.check_model(model, full_check=True)
    return ir.from_proto(model), weight


def test_unshared_initializer_is_not_modified(rng):
    model, _ = _model_with_shared_weight(rng, num_consumers=1)

    result = DuplicateInitializersPass()(model)

    assert result.modified is False
    assert len(result.model.graph.initializers) == 1


def test_two_consumers_produce_one_duplicate(rng):
    model, weight = _model_with_shared_weight(rng, num_consumers=2)

    result = DuplicateInitializersPass()(model)

    assert result.modified is True
    initializers = result.model.graph.initializers
    assert len(initializers) == 2
    assert "W" in initializers and "W_1" in initializers

    # The two MatMul nodes now point to different initializers.
    matmuls = [n for n in result.model.graph if n.op_type == "MatMul"]
    assert matmuls[0].inputs[1].name == "W"
    assert matmuls[1].inputs[1].name == "W_1"

    # Both initializers carry the same data.
    np.testing.assert_array_equal(initializers["W"].const_value.numpy(), weight)
    np.testing.assert_array_equal(initializers["W_1"].const_value.numpy(), weight)


def test_three_consumers_produce_two_duplicates(rng):
    model, _ = _model_with_shared_weight(rng, num_consumers=3)

    result = DuplicateInitializersPass()(model)

    assert result.modified is True
    initializers = result.model.graph.initializers
    assert set(initializers) == {"W", "W_1", "W_2"}

    matmuls = [n for n in result.model.graph if n.op_type == "MatMul"]
    assert [m.inputs[1].name for m in matmuls] == ["W", "W_1", "W_2"]


def test_skips_initializer_that_is_a_graph_output(rng):
    """Initializers exposed as graph outputs must not be duplicated."""
    model = onnx.parser.parse_model("""
        < ir_version: 10, opset_import: ["" : 21] >
        test_model (float[N, 4] X) => (float[N, 8] Y, float[4, 8] W)
        <float[4, 8] W>
        {
            Y = MatMul(X, W)
        }
    """)
    weight = rng.normal(size=(4, 8)).astype(np.float32)
    model.graph.initializer.extend([onnx.numpy_helper.from_array(weight, name="W")])
    ir_model = ir.from_proto(model)

    result = DuplicateInitializersPass()(ir_model)

    assert result.modified is False
    assert set(result.model.graph.initializers) == {"W"}


def test_forward_output_is_unchanged(rng):
    model, _ = _model_with_shared_weight(rng, num_consumers=2)
    original = model.clone(deep_copy=True)

    result = DuplicateInitializersPass()(model)
    assert result.modified is True

    samples = {
        "X0": rng.normal(size=(2, 4)).astype(np.float32),
        "X1": rng.normal(size=(2, 4)).astype(np.float32),
    }
    original_outputs, updated_outputs = onnx_forward_on_models(
        original, result.model, samples=samples
    )
    for orig, upd in zip(original_outputs, updated_outputs, strict=True):
        np.testing.assert_array_equal(orig, upd)

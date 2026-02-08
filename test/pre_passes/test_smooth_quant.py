import copy

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir._cloner import Cloner

from onnx_quantize import QConfig, SmoothQuantConfig
from onnx_quantize.core._qconfig import _SUPPORTED_OP_TYPES
from onnx_quantize.pre_passes import _add_qconfig_to_nodes, calibrate_model
from onnx_quantize.pre_passes.smooth_quant import SmoothQuantPass
from test.helpers import onnx_forward_on_models


@pytest.fixture(scope="module", autouse=True)
def patch_cloner_for_test():
    """
    Patch the Cloner class to perform a deep copy of the meta field.
    TODO: remove this if it is fixed in onnx_ir.
    """
    original_clone = Cloner.clone_node

    def clone_with_meta_copy(self, node):
        new_node = original_clone(self, node)

        if node.meta:
            for k, v in node.meta.items():
                new_node.meta[k] = copy.deepcopy(v)

            new_node.meta._invalid_keys = set(node.meta._invalid_keys)

        # Copy output properties
        for output, new_output in zip(node.outputs, new_node.outputs, strict=True):
            if output.meta:
                for k, v in output.meta.items():
                    new_output.meta[k] = copy.deepcopy(v)

                new_output.meta._invalid_keys = set(output.meta._invalid_keys)
        return new_node

    Cloner.clone_node = clone_with_meta_copy

    yield

    # Restore original after all tests in this module complete
    Cloner.clone_node = original_clone


def _preprocess_model(rng, model, calib_data):
    # Calibrate the model to compute activation scales for smooth quantization tests
    qconfig = QConfig(preprocessors=[SmoothQuantConfig()], calibration_data=calib_data)
    calibrate_model(
        model,
        qconfig,
    )

    _add_qconfig_to_nodes(model, qconfig)


def _get_matmul_model(rng, calib_data):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 21] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1>
                {
                    Y = MatMul(X, W1)
                }
            """)
    W1 = onnx.numpy_helper.from_array(rng.normal(size=(32, 64)).astype(np.float32), name="W1")
    model.graph.initializer.extend([W1])
    onnx.checker.check_model(model, full_check=True)
    model = ir.from_proto(model)
    _preprocess_model(rng, model, calib_data)

    return model


def _get_gemm_model(rng, calib_data):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64] B1, float[64, 128] W2>
                {
                    x1 = Gemm<transB=0>(X, W1, B1)
                    Y = Gemm(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(rng.normal(size=(32, 64)).astype(np.float32), name="W1")
    B1 = onnx.numpy_helper.from_array(
        rng.normal(size=(64,)).astype(np.float32),
        name="B1",
    )
    W2 = onnx.numpy_helper.from_array(rng.normal(size=(64, 128)).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, B1, W2])
    onnx.checker.check_model(model, full_check=True)
    model = ir.from_proto(model)
    _preprocess_model(rng, model, calib_data)

    return model


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "model_fn, expected_num_mul", [(_get_matmul_model, 1), (_get_gemm_model, 2)]
)
def test_smooth_quant(rng, alpha, model_fn, expected_num_mul):
    calib_data = rng.normal(size=(1, 32)).astype(np.float32)
    model = model_fn(rng, calib_data)

    smooth_quant_pass = SmoothQuantPass(alpha=alpha, target_op_types=_SUPPORTED_OP_TYPES)
    result = smooth_quant_pass(model.clone())

    # Check that the pass modified the model
    assert result.modified

    # Check that the modified model contains the expected nodes
    assert sum(node.op_type == "Mul" for node in result.model.graph) == expected_num_mul

    # Check that the modified model can be executed without errors
    samples = rng.normal(size=(1, 32)).astype(np.float32)

    original_outputs, updated_outputs = onnx_forward_on_models(
        model, result.model, samples={"X": samples}
    )
    np.testing.assert_allclose(original_outputs, updated_outputs, atol=5e-5)

    # Check that the input metadata is preserved correctly
    # When calibrating the updated model, we need to have the same activation
    # in node.meta["input"]
    updated_model_clone = result.model.clone()
    calibrate_model(
        updated_model_clone,
        QConfig(preprocessors=[SmoothQuantConfig()], calibration_data=calib_data),
    )

    for smooth_node, calib_node in zip(result.model.graph, updated_model_clone.graph, strict=True):
        if (activation := smooth_node.meta.get("input")) is not None and (
            calib_activation := calib_node.meta.get("input")
        ) is not None:
            np.testing.assert_allclose(
                activation,
                calib_activation,
                atol=1e-5,
            )

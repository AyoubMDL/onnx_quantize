import numpy as np
import onnx
import onnx_ir as ir
import pytest

from onnx_quantize import QConfig, SmoothQuantConfig
from onnx_quantize.pre_passes import _add_qconfig_to_nodes, calibrate_model
from onnx_quantize.pre_passes.smooth_quant import _SMOOTH_QUANT_OPS, SmoothQuantPass
from test.helpers import onnx_forward_on_models


def _preprocess_model(rng, model):
    # Calibrate the model to compute activation scales for smooth quantization tests
    calib_data = rng.normal(size=(1, 32)).astype(np.float32)
    qconfig = QConfig(preprocessors=[SmoothQuantConfig()], calibration_data=calib_data)
    calibrate_model(
        model,
        qconfig,
        op_types_to_calibrate=_SMOOTH_QUANT_OPS,
    )

    _add_qconfig_to_nodes(model, qconfig)


def _get_matmul_model(rng):
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
    _preprocess_model(rng, model)

    return model


def _get_gemm_model(rng):
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
    _preprocess_model(rng, model)

    return model


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "model_fn, expected_num_mul", [(_get_matmul_model, 1), (_get_gemm_model, 2)]
)
def test_smooth_quant(rng, alpha, model_fn, expected_num_mul):
    model = model_fn(rng)
    smooth_quant_pass = SmoothQuantPass(alpha=alpha)
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

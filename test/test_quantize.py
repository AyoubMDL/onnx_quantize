import numpy as np
import onnx
import onnx_ir as ir
import pytest

from onnx_quantize import GPTQConfig, QuantType, quantize
from onnx_quantize.core._qconfig import QActivationArgs, QConfig, QWeightArgs

from .helpers import onnx_forward_on_models


def _truncated_normal(rng, shape, scale=0.1, clip=2.5):
    x = rng.normal(0.0, scale, size=shape)
    return np.clip(x, -clip * scale, clip * scale).astype(np.float32)


def _get_matmul_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 21] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = MatMul(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_gemm_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[64, 32] W1, float[64] B1, float[64, 128] W2>
                {
                    x1 = Gemm<transB=1>(X, W1, B1)
                    Y = Gemm(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 32)), name="W1")
    B1 = onnx.numpy_helper.from_array(
        _truncated_normal(rng, (64,)),
        name="B1",
    )
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, B1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_matmul_add_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2, float[64] B1, float[128] B2>
                {
                    x1 = MatMul(X, W1)
                    x2 = Add(x1, B1)
                    x3 = MatMul(x2, W2)
                    Y = Add(x3, B2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    B1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64)), name="B1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    B2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (128)), name="B2")
    model.graph.initializer.extend([W1, B1, W2, B2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _test_quantize(model, qconfig, calibration_data=None):
    """Helper function to test quantization on a model."""
    # Convert to IR if needed
    is_proto = isinstance(model, onnx.ModelProto)
    if is_proto:
        original_model = model
        model = ir.from_proto(model)

    # Quantize model
    qmodel = quantize(model, qconfig)

    # Check type consistency
    assert isinstance(qmodel, type(model))

    # Convert to proto for further checks
    qmodel_proto = ir.to_proto(qmodel)
    model_proto = ir.to_proto(model) if not is_proto else original_model

    # Check all nodes are quantized (Assuming all ops are quantized)
    assert all(node.domain == "quant" for node in qmodel_proto.graph.node)

    # Check inference and compare outputs
    # Use calibration data if available, otherwise generate new test data
    if calibration_data is None:
        rng = np.random.default_rng(42)
        calibration_data = _truncated_normal(rng, (2, 32))

    original_output, quantized_output = onnx_forward_on_models(
        model_proto, qmodel_proto, samples={"X": calibration_data}
    )

    np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)


@pytest.mark.parametrize(
    "strategy, group_size",
    [
        ("tensor", None),
        ("channel", None),
        ("group", 16),
        ("group", 8),
    ],
)
@pytest.mark.parametrize(
    "dtype", [QuantType.QUInt8, QuantType.QInt8, QuantType.QUInt4, QuantType.QInt4]
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_only(rng, model_fn, strategy, group_size, dtype, symmetric, mse):
    model = model_fn(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy=strategy,
            group_size=group_size,
            symmetric=symmetric,
            mse=mse,
        )
    )

    _test_quantize(model, qconfig)


@pytest.mark.parametrize("strategy", ["tensor", "channel"])
@pytest.mark.parametrize(
    "dtype", [QuantType.QUInt8, QuantType.QInt8, QuantType.QInt4, QuantType.QUInt4]
)
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_only_gptq(rng, model_fn, strategy, dtype):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32))

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy=strategy,
            algorithm=GPTQConfig(block_size=16),
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_inputs(rng, model_fn, is_static, dtype, symmetric):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=symmetric,
        ),
        input_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_outputs(rng, model_fn, is_static, dtype):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=True,
        ),
        output_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_inputs_outputs(rng, model_fn, is_static, dtype, symmetric):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=symmetric,
        ),
        input_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        output_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(model, qconfig, calibration_data)


def test_no_quantization_needed(rng):
    model = _get_matmul_model(rng)

    qconfig = QConfig(
        weights=None,
        input_activations=None,
        output_activations=None,
    )

    qmodel = quantize(model, qconfig)

    # Check that the returned model is the same as the original
    assert qmodel == model

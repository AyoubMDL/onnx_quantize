import numpy as np
import onnx
import pytest

from onnx_quantize import GPTQConfig, QConfig, quantize

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


@pytest.mark.parametrize(
    "is_static, activations_dtype, activations_symmetric, weights_dtype, weights_symmetric",
    [
        (True, "uint8", False, "int8", True),
        (True, "int8", True, "int8", True),
        (True, "uint8", False, "int8", True),
        (False, "uint8", False, "uint8", False),
        (False, "uint8", False, "uint8", False),
        # 4bits (note it is only for weight only quantization)
        (False, "uint8", False, "uint4", False),
        (False, "uint8", False, "uint4", True),
        (False, "uint8", False, "int4", False),
        (False, "uint8", False, "int4", True),
    ],
)
@pytest.mark.parametrize(
    "weights_only, algorithm_config",
    [(True, None), (False, None), (True, GPTQConfig(block_size=16))],
)
@pytest.mark.parametrize(
    "strategy, group_size", [("tensor", None), ("channel", None), ("group", 4), ("group", 16)]
)
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize(
    rng,
    model_fn,
    is_static,
    weights_only,
    strategy,
    group_size,
    mse,
    activations_dtype,
    activations_symmetric,
    weights_dtype,
    weights_symmetric,
    algorithm_config,
):
    if strategy == "group" or weights_dtype in ("int4", "uint4"):
        # 4bits and group quantization only apply to weights
        weights_only = True

    if isinstance(algorithm_config, GPTQConfig) and strategy == "group":
        strategy = "channel"  # GPTQ only supports per-tensor/channel quantization
        group_size = None

    calibration_data = (
        _truncated_normal(rng, (2, 32))
        if not weights_only or isinstance(algorithm_config, GPTQConfig)
        else None
    )

    qconfig = QConfig(
        is_static=is_static,
        weights_only=weights_only,
        calibration_data=calibration_data,
        strategy=strategy,
        group_size=group_size,
        mse=mse,
        activations_dtype=activations_dtype,
        activations_symmetric=activations_symmetric,
        weights_dtype=weights_dtype,
        weights_symmetric=weights_symmetric,
        algorithm=algorithm_config,
    )

    model = model_fn(rng)
    qmodel = quantize(model, qconfig)

    # Check all nodes are quantized (Assumming all ops are quantized)
    assert all(node.domain == "quant" for node in qmodel.graph.node)

    # Check inference and compare outputs
    # Use calibration data if available, otherwise generate new test data
    data = calibration_data if calibration_data is not None else _truncated_normal(rng, (2, 32))
    original_output, quantized_output = onnx_forward_on_models(model, qmodel, samples={"X": data})

    np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)

import numpy as np
import onnx
import onnxruntime
import pytest

from onnx_quantize import QConfig, QuantType, quantize


def _get_matmul_model():
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = MatMul(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(32, 64).astype(np.float32), name="W1")
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_gemm_model():
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[64, 32] W1, float[64] B1, float[64, 128] W2>
                {
                    x1 = Gemm<transB=1>(X, W1, B1)
                    Y = Gemm(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(64, 32).astype(np.float32), name="W1")
    B1 = onnx.numpy_helper.from_array(
        np.random.randn(
            64,
        ).astype(np.float32),
        name="B1",
    )
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, B1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_matmul_add_model():
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
    W1 = onnx.numpy_helper.from_array(np.random.randn(32, 64).astype(np.float32), name="W1")
    B1 = onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), name="B1")
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    B2 = onnx.numpy_helper.from_array(np.random.randn(128).astype(np.float32), name="B2")
    model.graph.initializer.extend([W1, B1, W2, B2])
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize(
    "is_static, activations_dtype, activations_symmetric, weights_dtype, "
    "weights_symmetric, weights_per_channel",
    [
        (True, QuantType.QUInt8, False, QuantType.QInt8, True, False),
        (True, QuantType.QUInt8, True, QuantType.QInt8, True, False),
        (True, QuantType.QUInt8, False, QuantType.QInt8, True, True),
        (False, QuantType.QUInt8, False, QuantType.QUInt8, False, False),
    ],
)
@pytest.mark.parametrize("weights_only", [True, False])
@pytest.mark.parametrize("mse", [True, False])
@pytest.mark.parametrize("model", [_get_matmul_model(), _get_gemm_model(), _get_matmul_add_model()])
def test_quantize(
    model,
    is_static,
    weights_only,
    mse,
    activations_dtype,
    activations_symmetric,
    weights_dtype,
    weights_symmetric,
    weights_per_channel,
):
    qconfig = QConfig(
        is_static=is_static,
        weights_only=weights_only,
        mse=mse,
        activations_dtype=activations_dtype,
        activations_symmetric=activations_symmetric,
        weights_dtype=weights_dtype,
        weights_symmetric=weights_symmetric,
        weights_per_channel=weights_per_channel,
    )
    qmodel = quantize(model, qconfig)

    # Check all nodes are quantized (Assumming all ops are quantized)
    assert all(node.domain == "quant" for node in qmodel.graph.node)

    # Check inference
    qsession = onnxruntime.InferenceSession(qmodel.SerializeToString())

    rng = np.random.default_rng(99)
    data = rng.uniform(-1, 1, size=(2, 32)).astype(np.float32)
    qsession.run(None, {"X": data})

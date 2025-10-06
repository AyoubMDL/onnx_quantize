import numpy as np
import onnx
import onnxruntime
import pytest
from onnxruntime.quantization import QuantType

from onnx_quantize import QConfig, quantize


def _get_test_model():
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    x2 = Relu(x1)
                    Y = MatMul(x2, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(32, 64).astype(np.float32), name="W1")
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, W2])
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
def test_quantize(
    is_static,
    activations_dtype,
    activations_symmetric,
    weights_dtype,
    weights_symmetric,
    weights_per_channel,
):
    qconfig = QConfig(
        is_static=is_static,
        activations_dtype=activations_dtype,
        activations_symmetric=activations_symmetric,
        weights_dtype=weights_dtype,
        weights_symmetric=weights_symmetric,
        weights_per_channel=weights_per_channel,
    )
    qmodel = quantize(_get_test_model(), qconfig)

    # Check inference
    qsession = onnxruntime.InferenceSession(qmodel.SerializeToString())

    rng = np.random.default_rng(99)
    data = rng.uniform(-1, 1, size=(2, 32)).astype(np.float32)
    qsession.run(None, {"X": data})

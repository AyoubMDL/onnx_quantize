import numpy as np
import onnx
import onnx_ir as ir
import pytest

from onnx_quantize import OP_TYPES_TO_QUANTIZE, CalibrationParams, GPTQConfig, QConfig
from onnx_quantize.core._calibration.calibrate import calibrate_model


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
    return ir.from_proto(model)


@pytest.mark.parametrize("num_samples", [1, 10])
def test_calibrate_model_with_samples(num_samples):
    calibration_data = np.random.randn(num_samples, 32).astype(np.float32)
    qconfig = QConfig(is_static=True, calibration_data=calibration_data)
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_random_samples():
    qconfig = QConfig(is_static=True)
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_gptq():
    qconfig = QConfig(algorithm=GPTQConfig())
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input" in node.meta


@pytest.mark.parametrize("batch_size, num_samples", [(2, 10), (5, 10), (10, 10), (20, 10)])
def test_calibrate_model_with_batch_size(batch_size, num_samples):
    calibration_data = np.random.randn(num_samples, 32).astype(np.float32)
    qconfig = QConfig(
        is_static=True,
        calibration_data=calibration_data,
        calibration_params=CalibrationParams(batch_size=batch_size, num_samples=num_samples),
    )
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_batch_size_drops_excess_samples():
    calibration_data = np.random.randn(10, 32).astype(np.float32)
    qconfig = QConfig(
        is_static=True,
        calibration_data=calibration_data,
        calibration_params={
            "batch_size": 3,
            "num_samples": 10,
        },  # 10 // 3 = 3 batches, 1 sample dropped
    )
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that nodes have quantization params (should still work)
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_random_with_batch_size():
    """Test random calibration data generation with batch processing."""
    qconfig = QConfig(is_static=True, calibration_params={"batch_size": 4, "num_samples": 12})
    ir_model = calibrate_model(_get_test_model(), qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta

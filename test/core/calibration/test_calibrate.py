import numpy as np
import onnx
import onnx_ir as ir
import pytest

from onnx_quantize import OP_TYPES_TO_QUANTIZE, CalibrationParams, GPTQConfig
from onnx_quantize.core._calibration.calibrate import calibrate_model
from onnx_quantize.core._qconfig import QActivationArgs, QConfig, QuantType, QWeightArgs


def _truncated_normal(rng, shape, scale=0.1, clip=2.5):
    x = rng.normal(0.0, scale, size=shape)
    return np.clip(x, -clip * scale, clip * scale).astype(np.float32)


def _get_test_model(rng):
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
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, W2])
    return ir.from_proto(model)


@pytest.mark.parametrize("num_samples", [1, 10])
def test_calibrate_model_with_samples_inputs(rng, num_samples):
    calibration_data = _truncated_normal(rng, (num_samples, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_data=calibration_data,
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


@pytest.mark.parametrize("num_samples", [1, 10])
def test_calibrate_model_with_samples_outputs(rng, num_samples):
    calibration_data = _truncated_normal(rng, (num_samples, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_data=calibration_data,
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "output_scale" in node.meta
            assert "output_zero_point" in node.meta


@pytest.mark.parametrize("num_samples", [1, 10])
def test_calibrate_model_with_samples_inputs_outputs(rng, num_samples):
    calibration_data = _truncated_normal(rng, (num_samples, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_data=calibration_data,
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta
            assert "output_scale" in node.meta
            assert "output_zero_point" in node.meta


def test_calibrate_model_random_samples(rng):
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_gptq(rng):
    calibration_data = _truncated_normal(rng, (10, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8, algorithm=GPTQConfig()),
        calibration_data=calibration_data,
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input" in node.meta


@pytest.mark.parametrize("batch_size, num_samples", [(2, 10), (5, 10), (10, 10), (20, 10)])
def test_calibrate_model_with_batch_size(rng, batch_size, num_samples):
    calibration_data = _truncated_normal(rng, (num_samples, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_data=calibration_data,
        calibration_params=CalibrationParams(batch_size=batch_size, num_samples=num_samples),
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_batch_size_drops_excess_samples(rng):
    calibration_data = _truncated_normal(rng, (10, 32))
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_data=calibration_data,
        calibration_params={
            "batch_size": 3,
            "num_samples": 10,
        },  # 10 // 3 = 3 batches, 1 sample dropped
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that nodes have quantization params (should still work)
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta


def test_calibrate_model_random_with_batch_size(rng):
    """Test random calibration data generation with batch processing."""
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QUInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=True),
        calibration_params={"batch_size": 4, "num_samples": 12},
    )
    model = _get_test_model(rng)
    calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    # Check that expected nodes have quantization params
    for node in model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE:
            assert "input_scale" in node.meta
            assert "input_zero_point" in node.meta

import numpy as np
import pytest

from onnx_quantize.core._calibration.minmax import MinMaxCalibrator


def test_default_initialization():
    calibrator = MinMaxCalibrator()
    assert calibrator.momentum == 0.0
    assert calibrator.data == {}


def test_custom_momentum_initialization():
    calibrator = MinMaxCalibrator(momentum=0.5)
    assert calibrator.momentum == 0.5
    assert calibrator.data == {}


def test_momentum_near_upper_boundary():
    calibrator = MinMaxCalibrator(momentum=0.99)
    assert calibrator.momentum == 0.99


def test_invalid_momentum():
    with pytest.raises(AssertionError, match="Momentum must be in"):
        MinMaxCalibrator(momentum=1.0)

    with pytest.raises(AssertionError, match="Momentum must be in"):
        MinMaxCalibrator(momentum=1.5)

    with pytest.raises(AssertionError, match="Momentum must be in"):
        MinMaxCalibrator(momentum=-0.1)


def test_collect_single_batch():
    """Test collecting statistics from a single batch."""
    calibrator = MinMaxCalibrator()
    tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    calibrator.collect("test", tensor)

    assert "test" in calibrator.data
    assert calibrator.data["test"].min_val == 1.0
    assert calibrator.data["test"].max_val == 5.0

    min_val, max_val = calibrator.compute_range("test")
    # 0.0 is included in the range to have a valid zero point
    np.testing.assert_almost_equal(min_val, 0.0)
    np.testing.assert_almost_equal(max_val, 5.0)


def test_collect_negative_values():
    calibrator = MinMaxCalibrator()
    tensor = np.array([-5.0, -2.0, 0.0, 3.0, 7.0])

    calibrator.collect("test", tensor)

    assert calibrator.data["test"].min_val == -5.0
    assert calibrator.data["test"].max_val == 7.0

    min_val, max_val = calibrator.compute_range("test")
    np.testing.assert_almost_equal(min_val, -5.0)
    np.testing.assert_almost_equal(max_val, 7.0)


def test_collect_all_negative_values():
    calibrator = MinMaxCalibrator()
    tensor = np.array([-10.0, -5.0, -2.0, -1.0])

    calibrator.collect("test", tensor)

    assert calibrator.data["test"].min_val == -10.0
    assert calibrator.data["test"].max_val == -1.0

    min_val, max_val = calibrator.compute_range("test")
    # 0.0 is included in the range to have a valid zero point
    np.testing.assert_almost_equal(min_val, -10.0)
    np.testing.assert_almost_equal(max_val, 0.0)


def test_collect_multidimensional_array():
    calibrator = MinMaxCalibrator()
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 7.0, 2.5]])

    calibrator.collect("test", tensor)

    assert calibrator.data["test"].min_val == 0.5
    assert calibrator.data["test"].max_val == 7.0

    min_val, max_val = calibrator.compute_range("test")
    # 0.0 is included in the range to have a valid zero point
    np.testing.assert_almost_equal(min_val, 0.0)
    np.testing.assert_almost_equal(max_val, 7.0)


def test_collect_single_value():
    calibrator = MinMaxCalibrator()
    tensor = np.array([42.0])

    calibrator.collect("test", tensor)

    assert calibrator.data["test"].min_val == 42.0
    assert calibrator.data["test"].max_val == 42.0

    min_val, max_val = calibrator.compute_range("test")
    # 0.0 is included in the range to have a valid zero point
    np.testing.assert_almost_equal(min_val, 0.0)
    np.testing.assert_almost_equal(max_val, 42.0)


def test_collect_multiple_batches_no_momentum():
    calibrator = MinMaxCalibrator(momentum=0.0)

    calibrator.collect("test", np.array([1.0, 2.0, 3.0]))
    calibrator.collect("test", np.array([-0.5, 4.0, 2.5]))
    calibrator.collect("test", np.array([1.5, 3.5, 5.5]))

    # Should track absolute min and max across all batches
    assert calibrator.data["test"].min_val == -0.5
    assert calibrator.data["test"].max_val == 5.5

    min_val, max_val = calibrator.compute_range("test")
    np.testing.assert_almost_equal(min_val, -0.5)
    np.testing.assert_almost_equal(max_val, 5.5)


def test_collect_multiple_batches_with_momentum():
    calibrator = MinMaxCalibrator(momentum=0.8)

    # First batch: min=1.0, max=3.0
    calibrator.collect("test", np.array([-1.0, 2.0, 3.0]))
    assert calibrator.data["test"].min_val == -1.0
    assert calibrator.data["test"].max_val == 3.0

    # Second batch: min=-0.5, max=4.0
    # Expected: min = 0.8 * -1.0 + 0.2 * -0.5 = -0.9
    #           max = 0.8 * 3.0 + 0.2 * 4.0 = 3.2
    calibrator.collect("test", np.array([-0.5, 2.5, 4.0]))
    assert np.isclose(calibrator.data["test"].min_val, -0.9)
    assert np.isclose(calibrator.data["test"].max_val, 3.2)

    min_val, max_val = calibrator.compute_range("test")
    np.testing.assert_almost_equal(min_val, -0.9)
    np.testing.assert_almost_equal(max_val, 3.2)


def test_collect_multiple_tensors():
    calibrator = MinMaxCalibrator()

    calibrator.collect("tensor1", np.array([-1.5, 2.0, 3.0]))
    calibrator.collect("tensor2", np.array([-1.0, 0.0, 1.0]))
    calibrator.collect("tensor3", np.array([10.0, 20.0, 30.0]))

    assert len(calibrator.data) == 3
    assert calibrator.data["tensor1"].min_val == -1.5
    assert calibrator.data["tensor1"].max_val == 3.0
    assert calibrator.data["tensor2"].min_val == -1.0
    assert calibrator.data["tensor2"].max_val == 1.0
    assert calibrator.data["tensor3"].min_val == 10.0
    assert calibrator.data["tensor3"].max_val == 30.0


def test_collect_zero_array():
    calibrator = MinMaxCalibrator()
    tensor = np.zeros(10)

    calibrator.collect("test", tensor)

    assert calibrator.data["test"].min_val == 0.0
    assert calibrator.data["test"].max_val == 0.0


def test_compute_range_missing_tensor():
    calibrator = MinMaxCalibrator()

    with pytest.raises(KeyError, match="No calibration data collected for 'nonexistent'"):
        calibrator.compute_range("nonexistent")


def test_calibration_workflow(rng):
    calibrator = MinMaxCalibrator()
    # Simulate calibration across multiple batches
    batches = [
        rng.standard_normal((32, 64)).astype(np.float32),
        rng.standard_normal((32, 64)).astype(np.float32),
        rng.standard_normal((32, 64)).astype(np.float32),
    ]

    for _, batch in enumerate(batches):
        calibrator.collect("layer_output", batch)

    min_val, max_val = calibrator.compute_range("layer_output")

    # Verify range is computed
    assert isinstance(min_val, np.ndarray)
    assert isinstance(max_val, np.ndarray)
    assert np.all(min_val < max_val)


def test_multi_layer_calibration(rng):
    calibrator = MinMaxCalibrator()

    layers = ["conv1", "conv2", "fc1", "fc2", "output"]
    shapes = [(32, 64), (32, 128), (32, 256), (32, 512), (32, 10)]

    # Simulate one batch through all layers
    for layer_name, shape in zip(layers, shapes, strict=False):
        activation = rng.standard_normal(shape).astype(np.float32)
        calibrator.collect(layer_name, activation)

    # Verify all layers have calibration data
    assert len(calibrator.data) == len(layers)

    for layer_name in layers:
        min_val, max_val = calibrator.compute_range(layer_name)
        assert np.all(min_val < max_val)

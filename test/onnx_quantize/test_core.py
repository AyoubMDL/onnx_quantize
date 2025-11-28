import numpy as np
import pytest

from onnx_quantize import QuantType
from onnx_quantize.core import (
    QUANT_TYPE_TO_NP_DTYPE,
    calculate_mse_min_max,
    dequantize_tensor,
    get_quantization_params_from_tensor,
    get_quantized_range,
    quantize_bias,
    quantize_tensor,
)


@pytest.mark.parametrize(
    "quant_type, symmetric, expected",
    [
        # Signed QInt8 - Asymmetric
        (QuantType.QInt8, False, (-128, 127)),
        # Signed QInt8 - Symmetric
        (QuantType.QInt8, True, (-127, 127)),
        # Unsigned QUInt8 - Asymmetric
        (QuantType.QUInt8, False, (0, 255)),
        # Unsigned QUInt8 - Symmetric (not common, but supported)
        (QuantType.QUInt8, True, (0, 255)),
    ],
)
def test_get_quantized_range(quant_type, symmetric, expected):
    result = get_quantized_range(quant_type, symmetric)
    assert result == expected


@pytest.mark.parametrize(
    "fp_tensor, quant_type, symmetric",
    [
        # Asymmetric signed, 2D tensor for per-channel test
        (np.array([[-1.0, 11.0, 1.0], [-2.0, 3.0, 2.0]]), QuantType.QInt8, False),
        # Asymmetric unsigned, 2D tensor
        (np.array([[-2.0, 1.0, 2.0], [1.0, 2.0, 3.0]]), QuantType.QUInt8, False),
        # Symmetric signed, 2D tensor
        (np.array([[-2.0, -1.0, 1.0], [2.0, 1.0, -1.0]]), QuantType.QInt8, True),
        # Symmetric unsigned, 2D tensor
        (np.array([[-4.0, 5.0, 10.0], [10.0, 5.0, 0.0]]), QuantType.QUInt8, True),
    ],
)
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("mse", [False, True])
def test_get_quantization_params_valid(fp_tensor, quant_type, symmetric, per_channel, mse):
    # TODO: Rework this test
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, quant_type, symmetric, per_channel, mse
    )

    # scale is always positive
    assert np.all(scale >= 0)

    # zero_point must be integer or array of integers
    assert np.issubdtype(zero_point.dtype, np.integer)
    if per_channel:
        # For per_channel, zero_point and scale should be arrays with length == fp_tensor.shape[-1]
        assert scale.shape == (fp_tensor.shape[-1],)
        assert zero_point.shape == (fp_tensor.shape[-1],)
    else:
        assert np.isscalar(scale) and np.isscalar(zero_point)


def test_asymmetric_behavior_qint8():
    fp_tensor = np.array([-5.0, 0.0, 5.0])
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, QuantType.QInt8, is_symmetric=False, per_channel=False
    )

    # Expected range from get_quantized_range: (-128, 127)
    expected_scale = (5.0 - (-5.0)) / (127 - (-128))
    np.testing.assert_allclose(scale, expected_scale)
    assert -128 <= zero_point <= 127


def test_symmetric_behavior_qint8():
    fp_tensor = np.array([-10.0, -5.0, 5.0, 10.0])
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, QuantType.QInt8, is_symmetric=True, per_channel=False
    )

    expected_scale = 20.0 / 127
    np.testing.assert_allclose(scale, expected_scale)
    assert zero_point == 0


@pytest.mark.parametrize(
    "fp_tensor, quant_type, symmetric",
    [
        # Signed asymmetric
        (np.array([-1.0, 0.0, 1.0]), QuantType.QInt8, False),
        # Signed symmetric
        (np.array([-2.0, -1.0, 1.0, 2.0]), QuantType.QInt8, True),
        # Unsigned asymmetric
        (np.array([0.0, 1.0, 2.0]), QuantType.QUInt8, False),
        # Unsigned symmetric (rare but supported)
        (np.array([0.0, 5.0, 10.0]), QuantType.QUInt8, True),
    ],
)
def test_quantize_tensor_shapes_and_ranges(fp_tensor, quant_type, symmetric):
    q_tensor, scale, zero_point = quantize_tensor(fp_tensor, quant_type, symmetric)

    # Shape must match input
    assert q_tensor.shape == fp_tensor.shape

    # dtype must match quant_type
    assert q_tensor.dtype == QUANT_TYPE_TO_NP_DTYPE[quant_type]

    # Values must be inside quantized range
    qmin, qmax = get_quantized_range(quant_type, symmetric)
    assert np.all(q_tensor >= qmin)
    assert np.all(q_tensor <= qmax)

    # Scale must be non-negative
    assert scale >= 0

    # Check dequantization
    dq_tensor = dequantize_tensor(q_tensor, scale, zero_point)
    assert dq_tensor.shape == q_tensor.shape
    assert dq_tensor.dtype == np.float32


def test_quantize_bias(rng):
    bias = rng.random((16,)).astype(np.float32)
    input_scale = 1.5
    weight_scale = rng.random((16,)).astype(np.float32)
    q_bias, scale, zero_point = quantize_bias(bias, input_scale, weight_scale)

    # Shape must match input
    assert q_bias.shape == bias.shape
    np.testing.assert_array_equal(scale, input_scale * weight_scale)

    # dtype must match quant_type
    assert q_bias.dtype == np.int32
    assert zero_point == 0


@pytest.mark.parametrize(
    "per_channel, grid, patience",
    [
        (False, 50, 10),
        (True, 50, 10),
        (False, 5, 2),
        (False, 50, 1),
    ],
)
def test_calculate_mse_min_max(per_channel, grid, patience):
    """Test calculate_mse_min_max for shapes, ranges, and consistency."""
    fp_tensor = np.array([[-1.0, 2.0, 3.0], [0.5, -0.2, 1.0]], dtype=np.float32)
    best_min, best_max = calculate_mse_min_max(
        fp_tensor,
        quant_type=QuantType.QInt8,
        is_symmetric=False,
        per_channel=per_channel,
        grid=grid,
        patience=patience,
    )

    if per_channel:
        # Shape checks
        assert best_min.shape == (fp_tensor.shape[1],)
        assert best_max.shape == (fp_tensor.shape[1],)

        # Range checks
        min_vals = np.min(fp_tensor, axis=0)
        max_vals = np.max(fp_tensor, axis=0)
        assert np.all(best_min >= min_vals)
        assert np.all(best_max <= max_vals)
    else:
        assert np.isscalar(best_min) or best_min.shape == ()
        assert np.isscalar(best_max) or best_max.shape == ()

        assert best_min >= np.min(fp_tensor)
        assert best_max <= np.max(fp_tensor)

    # Consistency checks
    assert np.all(best_min <= best_max)
    assert np.isfinite(best_min).all()
    assert np.isfinite(best_max).all()

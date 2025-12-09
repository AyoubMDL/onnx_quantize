import numpy as np
import pytest

from onnx_quantize import QuantType
from onnx_quantize.core import (
    calculate_mse_min_max,
    dequantize_tensor,
    get_quantization_params_from_tensor,
    quantize_bias,
    quantize_tensor,
)


@pytest.mark.parametrize(
    "fp_tensor, quant_type, symmetric, expected_scale, expected_zp",
    [
        # Edge case: all zeros
        (np.array([0.0, 0.0, 0.0]), QuantType.QInt8, False, 1.0, -128),
        (np.array([0.0, 0.0, 0.0]), QuantType.QInt8, True, 1.0, 0),
        (np.array([0.0, 0.0, 0.0]), QuantType.QUInt8, False, 1.0, 0),
        # Edge case: single positive value
        (np.array([0.0, 0.0, 5.0]), QuantType.QInt8, False, 5.0 / 255, -128),
        (np.array([0.0, 0.0, 5.0]), QuantType.QInt8, True, 10.0 / 254, 0),
        # Edge case: max_val is 0, min_val is negative
        (np.array([-5.0, -2.0, 0.0]), QuantType.QInt8, False, 5.0 / 255, 127),
        (np.array([-5.0, -2.0, 0.0]), QuantType.QInt8, True, 5.0 / 127, 0),
        # Standard asymmetric signed
        (np.array([-5.0, 0.0, 5.0]), QuantType.QInt8, False, 10.0 / 255, 0),
        # Standard symmetric signed
        (np.array([-10.0, -5.0, 5.0, 10.0]), QuantType.QInt8, True, 10.0 / 127, 0),
        # Standard asymmetric unsigned
        (np.array([0.0, 5.0, 10.0]), QuantType.QUInt8, False, 10.0 / 255, 0),
        # Standard symmetric unsigned (with zero point != 0)
        (np.array([0.0, 5.0, 10.0]), QuantType.QUInt8, True, 20.0 / 255, 128),
    ],
)
def test_get_quantization_params_scalar(
    fp_tensor, quant_type, symmetric, expected_scale, expected_zp
):
    """Test get_quantization_params with scalar (non per-channel) cases."""
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, quant_type, symmetric, reduce_range=False, per_channel=False, mse=False
    )

    # Scale should be positive
    assert scale > 0
    assert scale.size == 1

    # Check expected scale
    np.testing.assert_allclose(scale, np.array(expected_scale, dtype=np.float32), rtol=1e-5)

    # Zero point should be scalar integer
    assert zero_point.dtype == quant_type.np_dtype
    assert zero_point.size == 1

    # Check zero point range or exact value
    np.testing.assert_allclose(zero_point, np.array(expected_zp, dtype=np.float32), rtol=1e-5)
    qmin, qmax = quant_type.qrange(symmetric)
    assert qmin <= zero_point <= qmax


@pytest.mark.parametrize(
    "fp_tensor, quant_type, symmetric",
    [
        # Per-channel with mixed signs
        (np.array([[-5.0, 0.0, 10.0], [-2.0, 5.0, 3.0]]), QuantType.QInt8, False),
        # Per-channel with all positive
        (np.array([[0.0, 5.0, 10.0], [1.0, 2.0, 3.0]]), QuantType.QUInt8, False),
        # Per-channel symmetric
        (np.array([[-10.0, -5.0, 5.0], [2.0, 1.0, -1.0]]), QuantType.QInt8, True),
        # Per-channel edge case: one channel all zeros
        (np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]), QuantType.QInt8, False),
    ],
)
@pytest.mark.parametrize("reduce_range", [False, True])
def test_get_quantization_params_per_channel(fp_tensor, quant_type, symmetric, reduce_range):
    """Test get_quantization_params with per-channel quantization."""
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, quant_type, symmetric, reduce_range, per_channel=True, mse=False
    )

    # Should return arrays with length equal to last dimension
    expected_len = fp_tensor.shape[-1]
    assert scale.shape == (expected_len,)
    assert zero_point.shape == (expected_len,)

    # All scales should be positive
    assert np.all(scale > 0)

    # Zero points should be integers
    assert zero_point.dtype == quant_type.np_dtype

    # Zero points should be within quantized range
    qmin, qmax = quant_type.qrange(symmetric)
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)


@pytest.mark.parametrize(
    "fp_tensor, symmetric",
    [
        # MSE with asymmetric
        (np.array([[-1.0, 11.0, 1.0], [-2.0, 3.0, 2.0]]), False),
        # MSE with symmetric
        (np.array([[-10.0, -5.0, 5.0], [2.0, 1.0, -1.0]]), True),
    ],
)
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
def test_get_quantization_params_mse(fp_tensor, quant_type, symmetric, per_channel):
    """Test get_quantization_params with MSE optimization."""
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor, quant_type, symmetric, reduce_range=False, per_channel=per_channel, mse=True
    )

    # Basic shape checks
    if per_channel:
        assert scale.shape == (fp_tensor.shape[-1],)
        assert zero_point.shape == (fp_tensor.shape[-1],)
    else:
        assert scale.size == 1
        assert zero_point.size == 1

    # Scale should be positive
    assert np.all(scale > 0)

    # Zero point should be integer and in range
    assert zero_point.dtype == quant_type.np_dtype
    qmin, qmax = quant_type.qrange(symmetric)
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)


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
@pytest.mark.parametrize("reduce_range", [False, True])
def test_quantize_tensor_shapes_and_ranges(fp_tensor, quant_type, symmetric, reduce_range):
    q_tensor, scale, zero_point = quantize_tensor(
        fp_tensor, quant_type, is_symmetric=symmetric, reduce_range=reduce_range
    )

    # Shape must match input
    assert q_tensor.shape == fp_tensor.shape

    # dtype must match quant_type
    assert q_tensor.dtype == quant_type.np_dtype

    # Values must be inside quantized range
    qmin, qmax = quant_type.qrange(symmetric)
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
@pytest.mark.parametrize("reduce_range", [False, True])
def test_calculate_mse_min_max(per_channel, grid, patience, reduce_range):
    """Test calculate_mse_min_max for shapes, ranges, and consistency."""
    fp_tensor = np.array([[-1.0, 2.0, 3.0], [0.5, -0.2, 1.0]], dtype=np.float32)
    best_min, best_max = calculate_mse_min_max(
        fp_tensor,
        quant_type=QuantType.QInt8,
        is_symmetric=False,
        reduce_range=reduce_range,
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

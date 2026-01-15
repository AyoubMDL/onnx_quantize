import math

import numpy as np
import pytest

from onnx_quantize import QuantizationStrategy, QuantType
from onnx_quantize.core._rtn import (
    _compute_min_max,
    _compute_min_max_mse,
    _compute_qparams_from_array,
    _dequantize_array,
    _preprocess_array,
    _quantize_array,
    _quantize_bias,
)


@pytest.mark.parametrize(
    "fp_array, quant_type, symmetric, expected_scale, expected_zp",
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
@pytest.mark.parametrize("mse", [False, True])
def test_get_quantization_params_scalar(
    fp_array, quant_type, symmetric, mse, expected_scale, expected_zp
):
    """Test get_quantization_params with scalar (non per-channel) cases."""
    scale, zero_point = _compute_qparams_from_array(
        fp_array,
        quant_type,
        QuantizationStrategy.TENSOR,
        group_size=-1,
        is_symmetric=symmetric,
        reduce_range=False,
        clip_ratio=1.0,
        mse=mse,
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
    "fp_array, quant_type, symmetric",
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
@pytest.mark.parametrize("mse", [False, True])
def test_get_quantization_params_per_channel(fp_array, quant_type, symmetric, mse):
    """Test get_quantization_params with per-channel quantization."""
    # fp_array shape: (out_channels, in_channels).
    # In practice, array comes in (in_channels, out_channels) shape, but
    # we transpose it in the quantization pipeline.
    scale, zero_point = _compute_qparams_from_array(
        fp_array,
        quant_type,
        QuantizationStrategy.CHANNEL,
        group_size=-1,
        is_symmetric=symmetric,
        reduce_range=False,
        clip_ratio=1.0,
        mse=mse,
    )

    # Should return arrays with length equal to last dimension
    expected_len = fp_array.shape[0]
    assert scale.shape == (expected_len, 1)
    assert zero_point.shape == (expected_len, 1)

    # All scales should be positive
    assert np.all(scale > 0)

    # Zero points should be integers
    assert zero_point.dtype == quant_type.np_dtype

    # Zero points should be within quantized range
    qmin, qmax = quant_type.qrange(symmetric)
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)


@pytest.mark.parametrize(
    "quant_type, symmetric, group_size",
    [
        # Group quantization with 2 groups
        (QuantType.QInt8, False, 2),
        # Group quantization with 4 groups
        (QuantType.QUInt8, False, 4),
        # Group quantization symmetric
        (QuantType.QInt8, True, 16),
        # Group quantization with one group all zeros
        (QuantType.QInt8, False, 7),
    ],
)
@pytest.mark.parametrize("mse", [False, True])
def test_get_quantization_params_group(quant_type, symmetric, group_size, mse):
    """Test get_quantization_params with group quantization."""
    fp_array = np.ones((32, 64), dtype=np.float32)
    in_channels, out_channels = fp_array.shape

    # Preprocess before computing qparams
    fp_array = _preprocess_array(fp_array, QuantizationStrategy.GROUP, group_size)

    scale, zero_point = _compute_qparams_from_array(
        fp_array,
        quant_type,
        QuantizationStrategy.GROUP,
        group_size=group_size,
        is_symmetric=symmetric,
        reduce_range=False,
        clip_ratio=1.0,
        mse=mse,
    )

    # Calculate expected number of groups
    num_groups = math.ceil(in_channels / group_size)

    # Check expected shapes
    assert scale.shape == (out_channels * num_groups, 1)
    assert zero_point.shape == (out_channels * num_groups, 1)

    # All scales should be positive
    assert np.all(scale > 0)

    # Zero points should be integers
    assert zero_point.dtype == quant_type.np_dtype

    # Zero points should be within quantized range
    qmin, qmax = quant_type.qrange(symmetric)
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)


def test_quantize_bias(rng):
    bias = rng.random((16,)).astype(np.float32)
    input_scale = 1.5
    weight_scale = rng.random((16,)).astype(np.float32)
    q_bias, scale, zero_point = _quantize_bias(bias, input_scale, weight_scale)

    # Shape must match input
    assert q_bias.shape == bias.shape
    np.testing.assert_array_equal(scale, input_scale * weight_scale)

    # dtype must match quant_type
    assert q_bias.dtype == np.int32
    assert zero_point == 0


@pytest.mark.parametrize(
    "grid, patience",
    [
        (50, 10),
        (50, 10),
        (5, 2),
        (50, 1),
    ],
)
@pytest.mark.parametrize("reduce_range", [False, True])
@pytest.mark.parametrize(
    "strategy, group_size",
    [
        (QuantizationStrategy.TENSOR, -1),
        (QuantizationStrategy.CHANNEL, -1),
        (QuantizationStrategy.GROUP, 16),
    ],
)
def test_calculate_mse_min_max(rng, grid, patience, reduce_range, strategy, group_size):
    """Test calculate_mse_min_max for shapes, ranges, and consistency."""
    fp_tensor = rng.standard_normal((32, 64), dtype=np.float32)

    original_min, original_max = _compute_min_max(
        fp_tensor, strategy=strategy, group_size=group_size
    )

    best_min, best_max = _compute_min_max_mse(
        fp_tensor,
        quant_type=QuantType.QInt8,
        strategy=strategy,
        group_size=group_size,
        is_symmetric=False,
        reduce_range=reduce_range,
        grid=grid,
        patience=patience,
    )

    assert best_min.shape == original_min.shape
    assert best_max.shape == original_max.shape
    assert np.all(best_min >= original_min)
    assert np.all(best_max <= original_max)

    assert np.all(best_min <= best_max)
    assert np.isfinite(best_min).all()
    assert np.isfinite(best_max).all()


@pytest.mark.parametrize(
    "quant_type, symmetric, reduce_range",
    [
        (QuantType.QInt8, False, False),
        (QuantType.QInt8, True, False),
        (QuantType.QUInt8, False, False),
        (QuantType.QUInt8, True, False),
        (QuantType.QInt8, False, True),
    ],
)
@pytest.mark.parametrize("mse", [False, True])
def test_quantize_array_tensor_strategy(rng, quant_type, symmetric, reduce_range, mse):
    """Test _quantize_array with TENSOR strategy (per-tensor quantization)."""
    # Create random array
    fp_array = rng.standard_normal((16, 32), dtype=np.float32)

    q_array, scale, zero_point = _quantize_array(
        fp_array,
        quant_type=quant_type,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        is_symmetric=symmetric,
        reduce_range=reduce_range,
        clip_ratio=1.0,
        mse=mse,
    )

    # Check output shapes
    assert q_array.shape == fp_array.shape
    assert scale.shape == ()
    assert zero_point.shape == ()

    # Check data types
    assert q_array.dtype == quant_type.np_dtype
    assert scale.dtype == np.float32
    assert zero_point.dtype == quant_type.np_dtype

    # Check quantized values are within range
    qmin, qmax = quant_type.qrange(symmetric, reduce_range)
    assert np.all(q_array >= qmin)
    assert np.all(q_array <= qmax)

    # Check scale is positive
    assert scale > 0

    # Check zero point is within range
    assert qmin <= zero_point <= qmax

    # Dequantize and check reconstruction
    dq_array = _dequantize_array(q_array, scale, zero_point)
    assert dq_array.shape == fp_array.shape
    assert dq_array.dtype == np.float32

    # The reconstruction error should be bounded by scale
    max_error = np.max(np.abs(dq_array - fp_array))
    assert max_error <= 2 * scale


@pytest.mark.parametrize(
    "quant_type, symmetric, reduce_range",
    [
        (QuantType.QInt8, False, False),
        (QuantType.QInt8, True, False),
        (QuantType.QUInt8, False, False),
        (QuantType.QUInt8, True, False),
        (QuantType.QInt8, False, True),
    ],
)
@pytest.mark.parametrize("mse", [False, True])
def test_quantize_array_channel_strategy(rng, quant_type, symmetric, reduce_range, mse):
    """Test _quantize_array with CHANNEL strategy (per-channel quantization)."""
    # Create random array
    in_channels, out_channels = 32, 64
    fp_array = rng.standard_normal((in_channels, out_channels), dtype=np.float32)

    q_array, scale, zero_point = _quantize_array(
        fp_array,
        quant_type=quant_type,
        strategy=QuantizationStrategy.CHANNEL,
        group_size=-1,
        is_symmetric=symmetric,
        reduce_range=reduce_range,
        clip_ratio=1.0,
        mse=mse,
    )

    # Check output shapes
    assert q_array.shape == fp_array.shape
    assert scale.shape == (out_channels,)
    assert zero_point.shape == (out_channels,)

    # Check data types
    assert q_array.dtype == quant_type.np_dtype
    assert scale.dtype == np.float32
    assert zero_point.dtype == quant_type.np_dtype

    # Check quantized values are within range
    qmin, qmax = quant_type.qrange(symmetric, reduce_range)
    assert np.all(q_array >= qmin)
    assert np.all(q_array <= qmax)

    # Check all scales are positive
    assert np.all(scale > 0)

    # Check all zero points are within range
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)

    # Dequantize and check reconstruction
    # For per-channel, need to broadcast scale and zero_point properly
    dq_array = _dequantize_array(
        q_array,
        scale,
        zero_point,
        preprocess=True,
        strategy=QuantizationStrategy.CHANNEL,
        group_size=-1,
    )

    assert dq_array.shape == fp_array.shape
    assert dq_array.dtype == np.float32

    # The reconstruction error should be bounded by scale
    max_error = np.max(np.abs(dq_array - fp_array))
    assert max_error <= 2 * scale.max()


@pytest.mark.parametrize(
    "quant_type, symmetric, reduce_range, group_size",
    [
        (QuantType.QInt8, False, False, 2),
        (QuantType.QInt8, False, False, 4),
        (QuantType.QUInt8, False, False, 8),
        (QuantType.QInt8, True, False, 16),
        (QuantType.QInt8, False, True, 8),
    ],
)
@pytest.mark.parametrize("mse", [False, True])
def test_quantize_array_group_strategy(rng, quant_type, symmetric, reduce_range, group_size, mse):
    """Test _quantize_array with GROUP strategy (group quantization)."""
    # Create random array
    in_channels, out_channels = 32, 64
    fp_array = rng.standard_normal((in_channels, out_channels), dtype=np.float32)

    q_array, scale, zero_point = _quantize_array(
        fp_array,
        quant_type=quant_type,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        is_symmetric=symmetric,
        reduce_range=reduce_range,
        clip_ratio=1.0,
        mse=mse,
    )

    # Check output shapes
    assert q_array.shape == fp_array.shape

    # Calculate expected number of groups
    num_groups = math.ceil(in_channels / group_size)
    expected_scale_shape = (out_channels * num_groups, 1)

    assert scale.shape == expected_scale_shape
    assert zero_point.shape == expected_scale_shape

    # Check data types
    assert q_array.dtype == quant_type.np_dtype
    assert scale.dtype == np.float32
    assert zero_point.dtype == quant_type.np_dtype

    # Check quantized values are within range
    qmin, qmax = quant_type.qrange(symmetric, reduce_range)
    assert np.all(q_array >= qmin)
    assert np.all(q_array <= qmax)

    # Check all scales are positive
    assert np.all(scale > 0)

    # Check all zero points are within range
    assert np.all(zero_point >= qmin)
    assert np.all(zero_point <= qmax)

    # Dequantize and check reconstruction
    # For per-channel, need to broadcast scale and zero_point properly
    dq_array = _dequantize_array(
        q_array,
        scale,
        zero_point,
        preprocess=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
    )

    assert dq_array.shape == fp_array.shape
    assert dq_array.dtype == np.float32

    # The reconstruction error should be bounded by scale
    max_error = np.max(np.abs(dq_array - fp_array))
    assert max_error <= 2 * scale.max()


def test_quantize_array_edge_case_all_zeros():
    """Test _quantize_array with all zeros."""
    fp_array = np.zeros((4, 4), dtype=np.float32)

    q_array, scale, zero_point = _quantize_array(
        fp_array,
        quant_type=QuantType.QInt8,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        is_symmetric=False,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # All quantized values should be at zero point
    assert np.all(q_array == zero_point)
    # Scale should be 1 (fallback value)
    assert scale == 1.0

    # Dequantize and verify
    dq_array = _dequantize_array(q_array, scale, zero_point)
    np.testing.assert_allclose(dq_array, fp_array, atol=1e-6)


def test_quantize_array_edge_case_single_value():
    """Test _quantize_array with single unique value."""
    fp_array = np.full((3, 3), 5.0, dtype=np.float32)

    q_array, scale, zero_point = _quantize_array(
        fp_array,
        quant_type=QuantType.QInt8,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        is_symmetric=False,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Dequantize and check we get close to original value
    dq_array = _dequantize_array(q_array, scale, zero_point)
    np.testing.assert_allclose(dq_array, fp_array, rtol=0.1)

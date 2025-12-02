import numpy as np
import pytest

from onnx_quantize import QuantType
from onnx_quantize.gptq import gptq_quantize


@pytest.fixture
def simple_weights(rng):
    """Create simple weight matrix for testing."""
    return rng.normal(0, 1, (16, 32)).astype(np.float32)


@pytest.fixture
def simple_inputs(rng):
    """Create simple input activations for testing."""
    return rng.normal(0, 1, (32, 16)).astype(np.float32)


@pytest.mark.parametrize("group_size", [8, 16, 32, 64, -1])
@pytest.mark.parametrize("block_size", [32, 64, 128, 256])
@pytest.mark.parametrize("percdamp", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("actorder", [True, False])
@pytest.mark.parametrize("mse", [True, False])
def test_gptq_quantize(
    simple_weights, simple_inputs, block_size, group_size, percdamp, actorder, mse
):
    w_q, w_scale, w_zero_point = gptq_quantize(
        simple_weights,
        simple_inputs,
        group_size=group_size,
        block_size=block_size,
        percdamp=percdamp,
        actorder=actorder,
        mse=mse,
    )

    # Check output shapes
    assert w_q.shape == simple_weights.shape
    assert isinstance(w_scale, np.ndarray)
    assert isinstance(w_zero_point, np.ndarray)

    # Check default dtypes (QInt8)
    assert w_q.dtype == np.int8
    assert w_scale.dtype == np.float32
    assert w_zero_point.dtype == np.int8

    # Check quantized values are in valid range for QInt8
    assert np.all(w_q >= -128)
    assert np.all(w_q <= 127)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
def test_quantization_types(simple_weights, simple_inputs, quant_type):
    """Test different quantization types."""
    w_q, w_scale, w_zero_point = gptq_quantize(simple_weights, simple_inputs, quant_type=quant_type)

    # Check correct dtype
    assert w_q.dtype == quant_type.np_dtype
    assert w_zero_point.dtype == quant_type.np_dtype

    # Check quantized values are in valid range
    if quant_type == QuantType.QInt8:
        assert np.all(w_q >= -128)
        assert np.all(w_q <= 127)
    elif quant_type == QuantType.QUInt8:
        assert np.all(w_q >= 0)
        assert np.all(w_q <= 255)


@pytest.mark.parametrize("reduce_range", [True, False])
@pytest.mark.parametrize("clip_ratio", [0.9, 0.95, 1.0])
def test_reduce_range_clip_ratio(simple_weights, simple_inputs, reduce_range, clip_ratio):
    w_q, _, _ = gptq_quantize(
        simple_weights, simple_inputs, reduce_range=reduce_range, clip_ratio=clip_ratio
    )

    assert w_q.shape == simple_weights.shape

    if reduce_range:
        # With reduced range for QInt8, values should be in [-64, 64]
        assert np.all(w_q >= -64)
        assert np.all(w_q <= 64)
    else:
        # Without reduced range for QInt8, values should be in [-128, 127]
        assert np.all(w_q >= -128)
        assert np.all(w_q <= 127)


@pytest.mark.parametrize("per_channel", [True, False])
def test_per_channel_quantization(simple_weights, simple_inputs, per_channel):
    """Test per-channel vs per-tensor quantization."""
    w_q, w_scale, w_zero_point = gptq_quantize(
        simple_weights, simple_inputs, per_channel=per_channel
    )

    assert w_q.shape == simple_weights.shape

    if per_channel and len(simple_weights.shape) > 1:
        # Per-channel should have scale/zp per output channel
        assert w_scale.size > 1
        assert w_zero_point.size > 1
    else:
        # Per-tensor should have scalar scale/zp
        assert w_scale.size == 1
        assert w_zero_point.size == 1

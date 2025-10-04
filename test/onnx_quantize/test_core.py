import numpy as np
import pytest
from onnxruntime.quantization import QuantType

from onnx_quantize.core import (
    QUANT_TYPE_TO_NP_DTYPE,
    get_quantization_params,
    get_quantized_range,
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
def test_get_quantization_params_valid(fp_tensor, quant_type, symmetric, per_channel):
    scale, zero_point = get_quantization_params(fp_tensor, quant_type, symmetric, per_channel)

    # scale is always positive
    assert np.all(scale >= 0)

    # zero_point must be integer or array of integers
    assert np.issubdtype(zero_point.dtype, np.integer)
    if per_channel:
        # For per_channel, zero_point and scale should be arrays with length == fp_tensor.shape[-1]
        assert scale.shape == (fp_tensor.shape[-1],)
        assert zero_point.shape == (fp_tensor.shape[-1],)
        if symmetric:
            assert np.all(zero_point == 0)
    else:
        if symmetric:
            assert zero_point == 0


def test_asymmetric_behavior_qint8():
    fp_tensor = np.array([-5.0, 0.0, 5.0])
    scale, zero_point = get_quantization_params(
        fp_tensor, QuantType.QInt8, is_symmetric=False, per_channel=False
    )

    # Expected range from get_quantized_range: (-128, 127)
    expected_scale = (5.0 - (-5.0)) / (127 - (-128))
    np.testing.assert_allclose(scale, expected_scale)
    assert -128 <= zero_point <= 127


def test_symmetric_behavior_qint8():
    fp_tensor = np.array([-10.0, -5.0, 5.0, 10.0])
    scale, zero_point = get_quantization_params(
        fp_tensor, QuantType.QInt8, is_symmetric=True, per_channel=False
    )

    # Expected: scale = max(abs(fp_tensor)) / 127
    expected_scale = 10.0 / 127
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

    assert zero_point == 0 if symmetric else True

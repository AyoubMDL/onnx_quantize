import numpy as np
import onnx
import pytest

from onnx_quantize import QuantizationStrategy, QuantType
from onnx_quantize.core import (
    _dequantize_array,
    _post_process_array,
    _preprocess_array,
    _quantize_array,
)
from onnx_quantize.qfunctions import QGemmDynamic8bits, QGemmStatic8bits, QGemmWeightsOnly
from onnx_quantize.qfunctions.qgemm import _make_qgemm_weight_only_grouped


def test_qgemm_static_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([[1, 0]], dtype=np.int32)
    x_scale = np.array([0.1], dtype=np.float32)
    w_scale = np.array([0.2], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QGemm
    result = QGemmStatic8bits(X, W, B, x_scale, w_scale, x_zero_point, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QGemmStatic8bits.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qgemm_dynamic_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([[1, 0]], dtype=np.float32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QGemm
    result = QGemmDynamic8bits(X, W, B, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QGemmDynamic8bits.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qgemm_weights_only_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([[1.0, 0.0]], dtype=np.float32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QGemm
    result = QGemmWeightsOnly(X, W, B, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QGemmWeightsOnly.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("strategy", [QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL])
def test_qmatmul_weights_only_outputs(rng, quant_type, is_symmetric, strategy):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 4)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(4, 4)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)

    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=strategy,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Run qfunction
    result = QGemmWeightsOnly(inputs, w_q, bias, w_scale, w_zero_point)
    expected_output = np.matmul(inputs, _dequantize_array(w_q, w_scale, w_zero_point)) + bias
    np.testing.assert_allclose(result, expected_output, atol=1e-5)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("group_size", [4, 16])
def test_qmatmul_weights_only_grouped_outputs(rng, quant_type, is_symmetric, group_size):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Run qfunction
    QGemmWeightsOnly8bitsGrouped = _make_qgemm_weight_only_grouped(group_size)
    result = QGemmWeightsOnly8bitsGrouped(
        inputs, w_q, bias, w_scale, w_zero_point, np.array(weights.T.shape, dtype=np.int64)
    )

    # Construct the numpy equivalent
    # w_q needs to be preprocessed before dequantization so it is compatible with scale and zp
    # Then post-processed after dequantization to restore original shape
    expected_output = (
        np.matmul(
            inputs,
            _post_process_array(
                _dequantize_array(
                    _preprocess_array(
                        w_q, strategy=QuantizationStrategy.GROUP, group_size=group_size
                    ),
                    w_scale,
                    w_zero_point,
                ),
                original_array=weights,
                strategy=QuantizationStrategy.GROUP,
                group_size=group_size,
            ),
        )
        + bias
    )

    # Compare with equivalent numpy ops
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Compare original with dequantized
    expected_output_full = np.matmul(inputs, weights) + bias
    np.testing.assert_allclose(result, expected_output_full, atol=5e-2)

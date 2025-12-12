import numpy as np
import onnx
import pytest

from onnx_quantize.core import QuantType, dequantize_tensor, quantize_tensor
from onnx_quantize.qfunctions import (
    QMatMulDynamic8bits,
    QMatMulStatic8bits,
    QMatMulWeightsOnly8bits,
)


def test_qmatmul_static_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    x_scale = np.array([0.1], dtype=np.float32)
    w_scale = np.array([0.2], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QMatMul
    result = QMatMulStatic8bits(X, W, x_scale, w_scale, x_zero_point, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QMatMulStatic8bits.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qmatmul_dynamic_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QMatMul
    result = QMatMulDynamic8bits(X, W, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QMatMulDynamic8bits.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qmatmul_weights_only_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QMatMul
    result = QMatMulWeightsOnly8bits(X, W, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QMatMulWeightsOnly8bits.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("per_channel", [True, False])
def test_qmatmul_weights_only_outputs(rng, quant_type, is_symmetric, per_channel):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 4)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(4, 4)).astype(np.float32)

    w_q, w_scale, w_zero_point = quantize_tensor(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        per_channel=per_channel,
    )

    # Run qfunction
    result = QMatMulWeightsOnly8bits(inputs, w_q, w_scale, w_zero_point)
    expected_output = np.matmul(inputs, dequantize_tensor(w_q, w_scale, w_zero_point))
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

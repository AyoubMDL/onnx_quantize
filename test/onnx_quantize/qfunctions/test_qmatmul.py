import numpy as np
import onnx

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

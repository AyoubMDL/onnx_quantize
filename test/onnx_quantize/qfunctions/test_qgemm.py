import numpy as np
import onnx

from onnx_quantize.qfunctions import QGemmDynamic8bits, QGemmStatic8bits, QGemmWeightsOnly8bits


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
    result = QGemmWeightsOnly8bits(X, W, B, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QGemmWeightsOnly8bits.to_function_proto()
    onnx.checker.check_function(function_proto)

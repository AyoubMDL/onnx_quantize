import numpy as np
import onnx
import pytest

from onnx_quantize import QuantizationStrategy, QuantType
from onnx_quantize.core import _dequantize_array, _quantize_array
from onnx_quantize.qfunctions._qlinear.qmatmul import QLinearMatMul


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_qlinear_matmul_weight_input_output_shape_and_type(dtype):
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=dtype)

    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=dtype)

    x_scale = np.array([0.1], dtype=np.float32)
    x_zero_point = np.array([0], dtype=dtype)

    out_scale = np.array([0.15], dtype=np.float32)
    out_zero_point = np.array([0], dtype=dtype)

    result = QLinearMatMul(
        X, W, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    # QLinearMatMul outputs quantized int8/uint8
    assert result.dtype == np.float32

    function_proto = QLinearMatMul.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qlinear_matmul_weight_input_output(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.CHANNEL,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Quantize inputs
    x_q, x_scale, x_zero_point = _quantize_array(
        inputs,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Compute expected intermediate output
    x_dequantized = _dequantize_array(x_q, x_scale, x_zero_point)
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    intermediate_output = np.matmul(x_dequantized, w_dequantized)

    # Quantize output
    out_q, out_scale, out_zero_point = _quantize_array(
        intermediate_output,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )
    expected_output = _dequantize_array(out_q, out_scale, out_zero_point)

    result = QLinearMatMul(
        inputs, w_q, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point
    )

    # Test against expected outputs
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Test against original outputs
    original_output = np.matmul(inputs, weights)
    np.testing.assert_allclose(result, original_output, atol=5e-2)

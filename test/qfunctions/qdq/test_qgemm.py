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
from onnx_quantize.core._qconfig import QActivationArgs, QConfig, QWeightArgs
from onnx_quantize.qfunctions._qdq.qgemm import (
    QGemmWeightDynamicInputOutputQDQ,
    QGemmWeightDynamicInputQDQ,
    QGemmWeightDynamicOutputQDQ,
    QGemmWeightInputOutputQDQ,
    QGemmWeightInputQDQ,
    QGemmWeightOutputQDQ,
    QGemmWeightsOnlyQDQ,
    _make_qgemm_weight_only_grouped,
    _make_qgemm_weight_only_grouped_4bits,
    qgemm_qdq_factory,
)


def test_qgemm_weights_only_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)

    result = QGemmWeightsOnlyQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightsOnlyQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("strategy", [QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL])
def test_qgemm_weights_only_qdq_outputs(rng, quant_type, is_symmetric, strategy):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

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

    # Quantize bias (using int32 for bias)
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    result = QGemmWeightsOnlyQDQ(inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point)

    # Simulate the operation
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    b_dequantized = _dequantize_array(b_q, b_scale, b_zero_point)
    expected_output = np.matmul(inputs, w_dequantized) + b_dequantized

    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Compare with original
    expected_output_full = np.matmul(inputs, weights) + bias
    np.testing.assert_allclose(result, expected_output_full, atol=5e-2)


def test_qgemm_weight_input_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)
    x_scale = np.array([0.1], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)

    result = QGemmWeightInputQDQ(
        X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, x_scale, x_zero_point
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightInputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_input_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
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

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    result = QGemmWeightInputQDQ(
        inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point, x_scale, x_zero_point
    )

    # Simulate the QDQ pattern
    x_dequantized = _dequantize_array(x_q, x_scale, x_zero_point)
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    b_dequantized = _dequantize_array(b_q, b_scale, b_zero_point)
    expected_output = np.matmul(x_dequantized, w_dequantized) + b_dequantized

    np.testing.assert_allclose(result, expected_output, atol=1e-5)


def test_qgemm_weight_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)
    out_scale = np.array([0.15], dtype=np.float32)
    out_zero_point = np.array([0], dtype=np.uint8)

    result = QGemmWeightOutputQDQ(
        X, W, B, w_scale, w_zero_point, b_scale, b_zero_point, out_scale, out_zero_point
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Compute expected output to determine output scale
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    b_dequantized = _dequantize_array(b_q, b_scale, b_zero_point)
    intermediate_output = np.matmul(inputs, w_dequantized) + b_dequantized

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

    result = QGemmWeightOutputQDQ(
        inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point, out_scale, out_zero_point
    )

    # Simulate the QDQ pattern
    expected_output = _dequantize_array(out_q, out_scale, out_zero_point)
    np.testing.assert_allclose(result, expected_output, atol=0.05)


def test_qgemm_weight_input_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)
    x_scale = np.array([0.1], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)
    out_scale = np.array([0.15], dtype=np.float32)
    out_zero_point = np.array([0], dtype=np.uint8)

    result = QGemmWeightInputOutputQDQ(
        X,
        W,
        B,
        w_scale,
        w_zero_point,
        b_scale,
        b_zero_point,
        x_scale,
        x_zero_point,
        out_scale,
        out_zero_point,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightInputOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_input_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
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

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Compute expected intermediate output
    x_dequantized = _dequantize_array(x_q, x_scale, x_zero_point)
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    b_dequantized = _dequantize_array(b_q, b_scale, b_zero_point)
    intermediate_output = np.matmul(x_dequantized, w_dequantized) + b_dequantized

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

    result = QGemmWeightInputOutputQDQ(
        inputs,
        w_q,
        b_q,
        w_scale,
        w_zero_point,
        b_scale,
        b_zero_point,
        x_scale,
        x_zero_point,
        out_scale,
        out_zero_point,
    )

    # Simulate the full QDQ pattern
    expected_output = _dequantize_array(out_q, out_scale, out_zero_point)
    np.testing.assert_allclose(result, expected_output, atol=0.05)


def test_qgemm_weight_dynamic_input_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)

    result = QGemmWeightDynamicInputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightDynamicInputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_dynamic_input_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    result = QGemmWeightDynamicInputQDQ(
        inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point
    )

    # The function internally does dynamic quantization of inputs
    expected_output = np.matmul(inputs, weights) + bias

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


def test_qgemm_weight_dynamic_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)

    result = QGemmWeightDynamicOutputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightDynamicOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_dynamic_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    result = QGemmWeightDynamicOutputQDQ(
        inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point
    )

    # The function internally does dynamic quantization of outputs
    expected_output = np.matmul(inputs, weights) + bias

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


def test_qgemm_weight_dynamic_input_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    B = np.array([1, 0], dtype=np.int32)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    b_scale = np.array([0.1], dtype=np.float32)
    b_zero_point = np.array([0], dtype=np.int32)

    result = QGemmWeightDynamicInputOutputQDQ(X, W, B, w_scale, w_zero_point, b_scale, b_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QGemmWeightDynamicInputOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qgemm_weight_dynamic_input_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)
    bias = rng.uniform(low=-1.0, high=1.0, size=(64,)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _quantize_array(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    result = QGemmWeightDynamicInputOutputQDQ(
        inputs, w_q, b_q, w_scale, w_zero_point, b_scale, b_zero_point
    )

    # The function internally does dynamic quantization of inputs and outputs
    expected_output = np.matmul(inputs, weights) + bias

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("group_size", [4, 16])
def test_qgemm_weights_only_grouped_outputs(rng, quant_type, is_symmetric, group_size):
    """Test grouped quantization for QGemm."""
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

    # Quantize bias
    b_q, b_scale, b_zero_point = _quantize_array(
        bias,
        quant_type=quant_type,
        is_symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
    )

    # Run qfunction
    QGemmWeightsOnlyGrouped = _make_qgemm_weight_only_grouped(group_size)
    result = QGemmWeightsOnlyGrouped(
        inputs,
        w_q,
        b_q,
        w_scale,
        w_zero_point,
        b_scale,
        b_zero_point,
        np.array(weights.T.shape, dtype=np.int64),
    )

    # Construct the numpy equivalent
    b_dequantized = _dequantize_array(b_q, b_scale, b_zero_point)
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
        + b_dequantized
    )

    # Compare with equivalent numpy ops
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Compare original with dequantized
    expected_output_full = np.matmul(inputs, weights) + bias
    np.testing.assert_allclose(result, expected_output_full, atol=5e-2)


def test_qgemm_weights_only_grouped_function_proto():
    group_size = 8
    QGemmWeightsOnlyGrouped = _make_qgemm_weight_only_grouped(group_size)

    function_proto = QGemmWeightsOnlyGrouped.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qgemm_weights_only_grouped_4bits_function_proto():
    group_size = 8
    QGemmWeightsOnlyGrouped4bits = _make_qgemm_weight_only_grouped_4bits(group_size)

    function_proto = QGemmWeightsOnlyGrouped4bits.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qgemm_qdq_factory_weights_only():
    qconfig = QConfig(weights=QWeightArgs(dtype=QuantType.QInt8))

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightsOnlyQDQ"


def test_qgemm_qdq_factory_static_input():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightInputQDQ"


def test_qgemm_qdq_factory_static_output():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightOutputQDQ"


def test_qgemm_qdq_factory_static_input_output():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
        output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightInputOutputQDQ"


def test_qgemm_qdq_factory_dynamic_input():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightDynamicInputQDQ"


def test_qgemm_qdq_factory_dynamic_output():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightDynamicOutputQDQ"


def test_qgemm_qdq_factory_dynamic_input_output():
    qconfig = QConfig(
        weights=QWeightArgs(dtype=QuantType.QInt8),
        input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
        output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightDynamicInputOutputQDQ"


def test_qgemm_qdq_factory_grouped():
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QInt8, strategy=QuantizationStrategy.GROUP, group_size=32
        )
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightsOnlyGrouped"


def test_qgemm_qdq_factory_grouped_4bits():
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QInt4, strategy=QuantizationStrategy.GROUP, group_size=32
        )
    )

    result = qgemm_qdq_factory(qconfig).__name__
    assert result == "QGemmWeightsOnlyGrouped"

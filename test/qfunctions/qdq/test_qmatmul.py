import numpy as np
import onnx
import pytest

from onnx_quantize import QuantizationStrategy, QuantType
from onnx_quantize.core._rtn import (
    _dequantize_array,
    _post_process_array,
    _preprocess_array,
    _rtn_quantize,
)
from onnx_quantize.qfunctions._qdq.qmatmul import (
    QMatMulWeightDynamicInputOutputQDQ,
    QMatMulWeightDynamicInputQDQ,
    QMatMulWeightDynamicOutputQDQ,
    QMatMulWeightsOnlyQDQ,
    QMatMulWeightStaticInputOutputQDQ,
    QMatMulWeightStaticInputQDQ,
    QMatMulWeightStaticOutputQDQ,
    _make_qmatmul_weight_only_grouped,
    _make_qmatmul_weight_only_grouped_4bits,
)


def test_qmatmul_weights_only_qdq_output_shape_and_type():
    # Create sample inputs
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    # Run QMatMul
    result = QMatMulWeightsOnlyQDQ(X, W, w_scale, w_zero_point)

    # Check type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    # Check function export
    function_proto = QMatMulWeightsOnlyQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("strategy", [QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL])
def test_qmatmul_weights_only_qdq_outputs(rng, quant_type, is_symmetric, strategy):
    # 4 bits dtype are not support in onnxscript eager execution
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=strategy,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Run qfunction
    result = QMatMulWeightsOnlyQDQ(inputs, w_q, w_scale, w_zero_point)
    expected_output = np.matmul(inputs, _dequantize_array(w_q, w_scale, w_zero_point))

    # Compare with equivalent numpy ops
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Compare original with dequantized
    expected_output_full = np.matmul(inputs, weights)
    np.testing.assert_allclose(result, expected_output_full, atol=5e-2)


def test_qmatmul_weight_static_input_qdq_output_shape_and_type():
    """Test that QMatMulWeightStaticInputQDQ produces correct output shape and type."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    x_scale = np.array([0.1], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightStaticInputQDQ(X, W, w_scale, w_zero_point, x_scale, x_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightStaticInputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_static_input_qdq_outputs(rng, quant_type, is_symmetric):
    """Test QMatMulWeightStaticInputQDQ numerical outputs."""
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Quantize inputs
    x_q, x_scale, x_zero_point = _rtn_quantize(
        inputs,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightStaticInputQDQ(inputs, w_q, w_scale, w_zero_point, x_scale, x_zero_point)

    # Simulate the QDQ pattern
    x_dequantized = _dequantize_array(x_q, x_scale, x_zero_point)
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    expected_output = np.matmul(x_dequantized, w_dequantized)

    np.testing.assert_allclose(result, expected_output, atol=1e-5)


def test_qmatmul_weight_static_output_qdq_output_shape_and_type():
    """Test that QMatMulWeightStaticOutputQDQ produces correct output shape and type."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    out_scale = np.array([0.15], dtype=np.float32)
    out_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightStaticOutputQDQ(X, W, w_scale, w_zero_point, out_scale, out_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightStaticOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_static_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Compute expected output to determine output scale
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    intermediate_output = np.matmul(inputs, w_dequantized)

    # Quantize output
    out_q, out_scale, out_zero_point = _rtn_quantize(
        intermediate_output,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightStaticOutputQDQ(
        inputs, w_q, w_scale, w_zero_point, out_scale, out_zero_point
    )

    # Simulate the QDQ pattern
    expected_output = _dequantize_array(out_q, out_scale, out_zero_point)
    # Use higher tolerance to account for numerical differences between numpy matmul and ONNX MatMul
    np.testing.assert_allclose(result, expected_output, atol=0.05)


def test_qmatmul_weight_static_input_output_qdq_output_shape_and_type():
    """Test that QMatMulWeightStaticInputOutputQDQ produces correct output shape and type."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)
    x_scale = np.array([0.1], dtype=np.float32)
    x_zero_point = np.array([0], dtype=np.uint8)
    out_scale = np.array([0.15], dtype=np.float32)
    out_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightStaticInputOutputQDQ(
        X, W, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightStaticInputOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_static_input_output_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Quantize inputs
    x_q, x_scale, x_zero_point = _rtn_quantize(
        inputs,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Compute expected intermediate output
    x_dequantized = _dequantize_array(x_q, x_scale, x_zero_point)
    w_dequantized = _dequantize_array(w_q, w_scale, w_zero_point)
    intermediate_output = np.matmul(x_dequantized, w_dequantized)

    # Quantize output
    out_q, out_scale, out_zero_point = _rtn_quantize(
        intermediate_output,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightStaticInputOutputQDQ(
        inputs, w_q, w_scale, w_zero_point, x_scale, x_zero_point, out_scale, out_zero_point
    )

    # Simulate the full QDQ pattern
    expected_output = _dequantize_array(out_q, out_scale, out_zero_point)
    # Use higher tolerance to account for numerical differences between numpy and ONNX ops
    np.testing.assert_allclose(result, expected_output, atol=0.05)


def test_qmatmul_weight_dynamic_input_qdq_output_shape_and_type():
    """Test that QMatMulWeightDynamicInputQDQ produces correct output shape and type."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightDynamicInputQDQ(X, W, w_scale, w_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightDynamicInputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_dynamic_input_qdq_outputs(rng, quant_type, is_symmetric):
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightDynamicInputQDQ(inputs, w_q, w_scale, w_zero_point)

    # The function internally does dynamic quantization of inputs
    # We can't predict the exact result without knowing the dynamic quantization behavior
    # But we can check that the result is reasonable
    expected_output = np.matmul(inputs, weights)

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


def test_qmatmul_weight_dynamic_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightDynamicOutputQDQ(X, W, w_scale, w_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightDynamicOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_dynamic_output_qdq_outputs(rng, quant_type, is_symmetric):
    """Test QMatMulWeightDynamicOutputQDQ numerical outputs."""
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightDynamicOutputQDQ(inputs, w_q, w_scale, w_zero_point)

    # The function internally does dynamic quantization of outputs
    expected_output = np.matmul(inputs, weights)

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


def test_qmatmul_weight_dynamic_input_output_qdq_output_shape_and_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    w_scale = np.array([0.2], dtype=np.float32)
    w_zero_point = np.array([0], dtype=np.uint8)

    result = QMatMulWeightDynamicInputOutputQDQ(X, W, w_scale, w_zero_point)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    function_proto = QMatMulWeightDynamicInputOutputQDQ.to_function_proto()
    onnx.checker.check_function(function_proto)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
def test_qmatmul_weight_dynamic_input_output_qdq_outputs(rng, quant_type, is_symmetric):
    """Test QMatMulWeightDynamicInputOutputQDQ numerical outputs."""
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    # Quantize weights
    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.TENSOR,
        group_size=-1,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    result = QMatMulWeightDynamicInputOutputQDQ(inputs, w_q, w_scale, w_zero_point)

    # The function internally does dynamic quantization of inputs and outputs
    expected_output = np.matmul(inputs, weights)

    # Check shape
    assert result.shape == expected_output.shape
    # Rough numerical check
    np.testing.assert_allclose(result, expected_output, atol=5e-2)


@pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize("group_size", [4, 16])
def test_qmatmul_weights_only_grouped_outputs(rng, quant_type, is_symmetric, group_size):
    """Test grouped quantization for QMatMul."""
    inputs = rng.uniform(low=-1.0, high=1.0, size=(2, 32)).astype(np.float32)
    weights = rng.uniform(low=-1.0, high=1.0, size=(32, 64)).astype(np.float32)

    w_q, w_scale, w_zero_point = _rtn_quantize(
        weights,
        quant_type=quant_type,
        is_symmetric=is_symmetric,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        reduce_range=False,
        clip_ratio=1.0,
        mse=False,
        scale_dtype=np.float32,
        zp_dtype=quant_type.np_dtype,
    )

    # Run qfunction
    QMatMulWeightsOnlyGrouped = _make_qmatmul_weight_only_grouped(group_size)
    result = QMatMulWeightsOnlyGrouped(
        inputs, w_q, w_scale, w_zero_point, np.array(weights.T.shape, dtype=np.int64)
    )

    # Construct the numpy equivalent
    # w_q needs to be preprocessed before dequantization so it is compatible with scale and zp
    # Then post-processed after dequantization to restore original shape
    expected_output = np.matmul(
        inputs,
        _post_process_array(
            _dequantize_array(
                _preprocess_array(w_q, strategy=QuantizationStrategy.GROUP, group_size=group_size),
                w_scale,
                w_zero_point,
            ),
            original_array=weights,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
        ),
    )

    # Compare with equivalent numpy ops
    np.testing.assert_allclose(result, expected_output, atol=1e-5)

    # Compare original with dequantized
    expected_output_full = np.matmul(inputs, weights)
    np.testing.assert_allclose(result, expected_output_full, atol=5e-2)


def test_qmatmul_weights_only_grouped_function_proto():
    group_size = 8
    QMatMulWeightsOnlyGrouped = _make_qmatmul_weight_only_grouped(group_size)

    function_proto = QMatMulWeightsOnlyGrouped.to_function_proto()
    onnx.checker.check_function(function_proto)


def test_qmatmul_weights_only_grouped_4bits_function_proto():
    group_size = 8
    QMatMulWeightsOnlyGrouped4bits = _make_qmatmul_weight_only_grouped_4bits(group_size)

    function_proto = QMatMulWeightsOnlyGrouped4bits.to_function_proto()
    onnx.checker.check_function(function_proto)

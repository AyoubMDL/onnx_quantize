__all__ = ["_rtn_quantize"]


import numpy as np

from onnx_quantize.core._algorithms.utils import (
    _compute_qparams_from_array,
    _post_process_array,
    _preprocess_array,
    _quantize_array_from_qparams,
)
from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QuantizationStrategy


def _rtn_quantize(
    array: np.ndarray,
    quant_type: QuantType,
    strategy: QuantizationStrategy,
    group_size: int,
    is_symmetric: bool,
    reduce_range: bool,
    clip_ratio: float,
    mse: bool,
    scale_dtype: np.dtype,
    zp_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes a tensor using asymmetric quantization.

    Args:
        array (np.ndarray): The floating-point tensor to quantize.
        quant_type (QuantType): The quantization type.
        strategy (QuantizationStrategy): The quantization strategy to use.
        group_size (int): The group size for group quantization.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.
        per_channel (bool): Whether to perform per-channel quantization.
        clip_ratio (float): percentile of clip.
        mse (bool): Whether to use MSE minimization to compute quantization parameters.
        scale_dtype (np.dtype): The desired data type for the scale.
        zp_dtype (np.dtype): The desired data type for the zero point.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The quantized tensor, scale, and zero-point.
    """
    preprocessed_array = _preprocess_array(array, strategy, group_size)
    scale, zero_point = _compute_qparams_from_array(
        preprocessed_array,
        quant_type,
        strategy,
        group_size,
        is_symmetric,
        reduce_range,
        clip_ratio=clip_ratio,
        mse=mse,
        scale_dtype=scale_dtype,
        zp_dtype=zp_dtype,
    )
    q_tensor = _quantize_array_from_qparams(
        preprocessed_array, scale, zero_point, quant_type, is_symmetric, reduce_range
    )

    # Squeeze scale and zero_point to remove unnecessary dimensions (ort constraint)
    # For group quantization, extra dimension is needed
    if strategy in {QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL}:
        scale, zero_point = np.squeeze(scale), np.squeeze(zero_point)

    # Reshape to original
    post_processed_qarray = _post_process_array(q_tensor, array, strategy, group_size)

    return post_processed_qarray, scale, zero_point


def _quantize_bias(bias, input_scale, weight_scale):
    """Linear quantization for single bias tensor quantized_bias = fp_bias / bias_scale.

    Args:
        bias (np.ndarray): bias weight to be quantized
        weight_scale: [float or torch.FloatTensor] weight scale tensor
        input_scale: [float] input scale

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The quantized tensor, scale, and zero-point.
    """
    assert bias.ndim == 1
    assert bias.dtype == np.float32
    assert np.size(input_scale) == 1
    assert weight_scale.dtype == np.float32
    assert weight_scale.size == 1 or bias.size == weight_scale.size

    bias_scale = weight_scale * input_scale
    qbias = _quantize_array_from_qparams(
        bias,
        scale=bias_scale,
        zero_point=0,
        quant_type=QuantType.QInt32,
        is_symmetric=False,
        reduce_range=False,
    )
    return qbias, bias_scale, 0

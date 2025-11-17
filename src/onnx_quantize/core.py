import dataclasses
import enum
import warnings

import numpy as np


class QuantType(enum.Enum):
    """Enumeration of quantization types."""

    QInt8 = 0
    QUInt8 = 1
    QInt32 = 2


QUANT_TYPE_TO_NP_DTYPE = {
    QuantType.QInt8: np.int8,
    QuantType.QUInt8: np.uint8,
    QuantType.QInt32: np.int32,
}

_SIGNED_QUANT_TYPES = {QuantType.QInt8}


@dataclasses.dataclass
class QConfig:
    """QnConfig is the configuration class handling all the quantization parameters.

    Args:
        is_static (`bool`, , defaults to `True`): Whether it is static or dynamic quantization.
        weights_only (`bool`, , defaults to `False`): Whether to quantize only weights or not.
        activations_dtype (`QuantType`, defaults to `QuantType.QUInt8`):
            The quantization data types to use for the activations.
        activations_symmetric (`bool`, defaults to `False`):
            Whether to apply symmetric quantization on the activations.
        weights_dtype (`QuantType`, defaults to `QuantType.QInt8`):
            The quantization data types to use for the weights.
        weights_symmetric (`bool`, defaults to `True`):
            Whether to apply symmetric quantization on the weights.
        weights_per_channel (`bool`, defaults to `False`):
            Whether we should quantize per-channel (also known as "per-row"). Enabling this
            can increase overall accuracy while making the quantized model heavier.
            For activation, onnx has weird per channel ops for the activations.
    """

    is_static: bool = True
    weights_only: bool = False
    calibration_data: np.ndarray | None = None
    activations_dtype: QuantType = QuantType.QUInt8
    activations_symmetric: bool = False
    weights_dtype: QuantType = QuantType.QInt8
    weights_symmetric: bool = True
    weights_per_channel: bool = False

    def __post_init__(self):
        """Check: can't use dynamic quantization with int8 weights."""
        if not self.is_static and self.weights_dtype == QuantType.QInt8:
            raise ValueError(
                "Dynamic quantization cannot be used with int8 weights. "
                "Please set weights_dtype=QuantType.QUInt8 or use static quantization."
            )
        if self.weights_only and self.calibration_data is not None:
            warnings.warn(
                "calibration_data is ignored when weight_only is set to True.", stacklevel=1
            )


def get_quantized_range(quant_type=QuantType.QInt8, is_symmetric=False):
    """Computes the minimum and maximum representable values for asymmetric quantization.

    Args:
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8). Defaults to QuantType.QInt8.
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.

    Returns:
        tuple[int, int]: The minimum and maximum quantized values.
    """
    bitwidth = np.dtype(QUANT_TYPE_TO_NP_DTYPE[quant_type]).itemsize * 8

    if quant_type in _SIGNED_QUANT_TYPES:
        bitwidth -= 1
        quantized_min = -(1 << (bitwidth))
        if is_symmetric:
            quantized_min += 1

        quantized_max = (1 << (bitwidth)) - 1

    else:
        # For symmetric + unsinged is not commonly used, but we can still define:
        # 0 .. (2^bitwidth - 1)
        quantized_min = 0
        quantized_max = (1 << bitwidth) - 1

    return quantized_min, quantized_max


def get_quantization_params(fp_tensor, quant_type, is_symmetric, per_channel):
    """Calculates the quantization parameters.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to be quantized.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8)
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.
        per_channel (bool): Whether to compute per-channel quantization parameters.

    Returns:
        tuple[np.ndarray, np.ndarray]: The quantization scale factor and null zero point.
    """

    def _get_quantization_params_asymmetric(fp_tensor, quant_type, axis):
        quantized_min, quantized_max = get_quantized_range(quant_type, is_symmetric=False)
        r_min, r_max = np.min(fp_tensor, axis=axis), np.max(fp_tensor, axis=axis)

        scale = (r_max - r_min) / (quantized_max - quantized_min)
        zero_point = quantized_min - (r_min / scale)
        zero_point = np.round(np.clip(zero_point, quantized_min, quantized_max)).astype(np.int8)

        return scale.astype(np.float32), zero_point.astype(QUANT_TYPE_TO_NP_DTYPE[quant_type])

    def _get_quantization_params_symmetric(fp_tensor, quant_type, axis):
        _, quantized_max = get_quantized_range(quant_type, is_symmetric=True)
        r_max = np.max(np.abs(fp_tensor), axis=axis)
        scale = r_max / quantized_max

        return scale.astype(np.float32), np.zeros_like(
            scale, dtype=QUANT_TYPE_TO_NP_DTYPE[quant_type]
        )

    axis = 0 if per_channel else None
    if is_symmetric:
        return _get_quantization_params_symmetric(fp_tensor, quant_type, axis)
    return _get_quantization_params_asymmetric(fp_tensor, quant_type, axis)


def _linear_quantize(fp_tensor, quant_type, is_symmetric, scale, zero_point):
    fp_tensor_scaled = fp_tensor / scale
    shifted_tensor = np.round(fp_tensor_scaled).astype(np.int32) + zero_point

    quantized_min, quantized_max = get_quantized_range(quant_type, is_symmetric)
    q_tensor = np.clip(shifted_tensor, quantized_min, quantized_max)
    q_tensor = q_tensor.astype(QUANT_TYPE_TO_NP_DTYPE[quant_type])

    return q_tensor


def quantize_tensor(fp_tensor, quant_type=QuantType.QInt8, is_symmetric=False, per_channel=False):
    """Quantizes a tensor using asymmetric quantization.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to quantize.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8). Defaults to QuantType.QInt8.
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.
        per_channel (bool): Whether to perform per-channel quantization. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The quantized tensor, scale, and zero-point.
    """
    scale, zero_point = get_quantization_params(fp_tensor, quant_type, is_symmetric, per_channel)
    q_tensor = _linear_quantize(fp_tensor, quant_type, is_symmetric, scale, zero_point)
    return q_tensor, scale, zero_point


def dequantize_tensor(q_tensor, scale, zero_point):
    """Dequantizes a tensor.

    Args:
        q_tensor (np.ndarray): The quantized tensor to dequantize.
        scale (np.ndarray): The scaling factor.
        zero_point (np.ndarray): The zero point.

    Returns:
        np.ndarray: The dequantized tensor
    """
    return (q_tensor - zero_point).astype(np.float32) * scale


def quantize_bias(bias, input_scale, weight_scale):
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
    qbias = _linear_quantize(
        bias, QuantType.QInt32, is_symmetric=False, scale=bias_scale, zero_point=0
    )
    return qbias, bias_scale, 0

import dataclasses

import numpy as np
from onnxruntime.quantization import QuantType


QUANT_TYPE_TO_NP_DTYPE = {
    QuantType.QInt8: np.int8,
    QuantType.QUInt8: np.uint8,
}

_SIGNED_QUANT_TYPES = {QuantType.QInt8}


@dataclasses.dataclass
class QConfig:
    """QnConfig is the configuration class handling all the quantization parameters.

    Args:
        is_static (`bool`, , defaults to `True`): Whether it is static or dynamic quantization.
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

    # Linear quantization
    # if np.size(scale) != 1:
    #     scale = scale[:, None]
    #     zero_point = zero_point[:, None]

    fp_tensor_scaled = fp_tensor / scale
    shifted_tensor = np.round(fp_tensor_scaled).astype(np.int32) + zero_point

    quantized_min, quantized_max = get_quantized_range(quant_type, is_symmetric)
    q_tensor = np.clip(shifted_tensor, quantized_min, quantized_max)
    q_tensor = q_tensor.astype(QUANT_TYPE_TO_NP_DTYPE[quant_type])

    return q_tensor, scale, zero_point

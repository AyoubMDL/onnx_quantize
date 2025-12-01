import dataclasses
import enum
import warnings

import numpy as np


_DTYPE_RANGES = {
    np.uint8: (0, 255),
    np.int8: (-128, 127),
    np.uint32: (0, 2**32 - 1),
    np.int32: (-(2**31), 2**31 - 1),
}

_SYMMETRIC_RANGES = {
    np.int8: (-127, 127),
    np.int32: (-(2**31 - 1), 2**31 - 1),
}

_REDUCED_RANGES = {
    np.uint8: (0, 127),
    np.int8: (-64, 64),
    np.uint32: (0, 2**31 - 1),
    np.int32: (-(2**30), 2**30),
}


class QuantType(enum.Enum):
    """Enumeration of quantization types."""

    QInt8 = 0
    QUInt8 = 1
    QInt32 = 2
    QUInt32 = 3

    @property
    def np_dtype(self):
        if self == QuantType.QInt8:
            return np.int8
        if self == QuantType.QUInt8:
            return np.uint8
        if self == QuantType.QInt32:
            return np.int32
        if self == QuantType.QUInt32:
            return np.uint32

    def qrange(self, is_symmetric, reduce_range=False):
        np_dtype = self.np_dtype
        if reduce_range:
            qrange = _REDUCED_RANGES.get(np_dtype)
        elif is_symmetric and np_dtype in _SYMMETRIC_RANGES:
            qrange = _SYMMETRIC_RANGES[np_dtype]
        else:
            qrange = _DTYPE_RANGES.get(np_dtype)

        return qrange


@dataclasses.dataclass
class QConfig:
    """QnConfig is the configuration class handling all the quantization parameters.

    Args:
        is_static (`bool`, , defaults to `True`): Whether it is static or dynamic quantization.
        weights_only (`bool`, , defaults to `False`): Whether to quantize only weights or not.
        clip_ratio (float, optional): percentile of clip. Defaults to 1.0
        reduce_range (bool, optional): Whether to use reduced range for quantization.
            Defaults to False.
        mse (`bool`, , defaults to `False`): Whether to use MSE minimization to compute
            quantization parameters.
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
    clip_ratio: float = 1.0
    reduce_range: bool = False
    mse: bool = False
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


def calculate_mse_min_max(
    fp_tensor,
    quant_type,
    is_symmetric,
    reduce_range,
    per_channel,
    maxshrink=0.20,
    patience=5,
    grid=100.0,
    norm=2.4,
):
    """Calculates the optimal min and max values for quantization using MSE minimization.

    This function searches for the best quantization range by iteratively shrinking
    the min/max values and evaluating the mean squared error between the original
    and quantized tensors. It uses early stopping to avoid unnecessary iterations.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to be quantized.
        quant_type (QuantType): The quantization data type.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.
        per_channel (bool): Whether to perform per-channel quantization or
            per-tensor quantization.
        maxshrink (float, optional): Maximum shrinkage factor as a fraction of the
            search grid. Defaults to 0.20.
        patience (int, optional): Number of iterations without improvement before
            early stopping. Defaults to 5.
        grid (float, optional): Number of grid points to search in the shrinkage
            range. Defaults to 100.0.
        norm (float, optional): The norm to use for error calculation (Lp norm).
            Defaults to 2.4.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - best_min_val: The optimal minimum value(s) for quantization.
            - best_max_val: The optimal maximum value(s) for quantization.
            For per_channel=True, these are arrays; for per_channel=False, scalars.
    """
    axis = 0 if per_channel else None
    min_val = np.amin(fp_tensor, axis=axis)
    max_val = np.amax(fp_tensor, axis=axis)

    best_error = np.full_like(min_val, np.finfo(min_val.dtype).max)
    best_min_val = min_val.copy()
    best_max_val = max_val.copy()

    # Early stopping params
    no_improve_count = 0

    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrinked_min_val = p * min_val
        shrinked_max_val = p * max_val

        candidate_scales, candidate_zero_points = get_quantization_params(
            min_vals=shrinked_min_val,
            max_vals=shrinked_max_val,
            quant_type=quant_type,
            is_symmetric=is_symmetric,
            reduce_range=reduce_range,
        )
        q = fake_quantize_tensor(
            fp_tensor,
            candidate_scales,
            candidate_zero_points,
            quant_type,
            is_symmetric,
            reduce_range,
        )

        q -= fp_tensor
        q = np.abs(q)
        q = np.power(q, norm)

        err = np.sum(q, axis=axis)

        tmp = err < best_error

        if per_channel:
            # Vector case: boolean mask indexing
            if np.any(tmp):
                best_error[tmp] = err[tmp]
                best_min_val[tmp] = shrinked_min_val[tmp]
                best_max_val[tmp] = shrinked_max_val[tmp]
            else:
                no_improve_count += 1
        else:
            # Scalar case
            if tmp:
                best_error = err
                best_min_val = shrinked_min_val
                best_max_val = shrinked_max_val
            else:
                no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            break

    return best_min_val, best_max_val


def get_quantization_params(min_vals, max_vals, quant_type, is_symmetric, reduce_range):
    """Calculates the quantization parameters.

    Args:
        min_vals (np.ndarray): The minimum values of the tensor to be quantized.
        max_vals (np.ndarray): The maximum values of the tensor to be quantized.
        quant_type (QuantType, optional): The quantization type.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.

    Returns:
        tuple[np.ndarray, np.ndarray]: The quantization scale factor and null zero point.
    """

    def _get_quantization_params_asymmetric(min_vals, max_vals, quant_type):
        quantized_min, quantized_max = quant_type.qrange(
            is_symmetric=False, reduce_range=reduce_range
        )

        scale = (max_vals - min_vals) / (quantized_max - quantized_min)
        zero_point = quantized_min - (min_vals / scale)
        zero_point = np.round(np.clip(zero_point, quantized_min, quantized_max))

        return scale.astype(np.float32), zero_point.astype(quant_type.np_dtype)

    def _get_quantization_params_symmetric(max_vals, quant_type):
        quantized_min, quantized_max = quant_type.qrange(
            is_symmetric=True, reduce_range=reduce_range
        )
        scale = (2 * max_vals) / (quantized_max - quantized_min)
        zero = np.multiply(
            np.ones(max_vals.shape), np.round((quantized_max + quantized_min) / 2.0)
        ).astype(quant_type.np_dtype)

        return scale.astype(np.float32), zero

    if is_symmetric:
        max_vals = np.maximum(np.abs(min_vals), np.abs(max_vals))
        return _get_quantization_params_symmetric(max_vals, quant_type)
    return _get_quantization_params_asymmetric(min_vals, max_vals, quant_type)


def get_quantization_params_from_tensor(
    fp_tensor,
    quant_type,
    is_symmetric=False,
    reduce_range=False,
    per_channel=True,
    clip_ratio=1.0,
    mse=False,
):
    """Calculates the quantization parameters from a tensor.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to be quantized.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8)
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.
        reduce_range (bool, optional): Whether to use reduced range for quantization.
            Defaults to False.
        per_channel (bool): Whether to compute per-channel quantization parameters.
            Defaults to True.
        clip_ratio (float, optional): percentile of clip. Defaults to 1.0
        mse (bool, optional): Whether to use MSE minimization to compute quantization parameters.
            Defaults to False

    Returns:
        tuple[np.ndarray, np.ndarray]: The quantization scale factor and null zero point.
    """
    axis = 0 if per_channel else None
    min_vals, max_vals = (
        np.min(fp_tensor, axis=axis) * clip_ratio,
        np.max(fp_tensor, axis=axis) * clip_ratio,
    )

    # Include Zero in the range to have a valid zero point
    min_vals = np.minimum(min_vals, 0)
    max_vals = np.maximum(max_vals, 0)

    if mse:
        min_vals, max_vals = calculate_mse_min_max(
            fp_tensor, quant_type, is_symmetric, reduce_range, per_channel
        )

    return get_quantization_params(min_vals, max_vals, quant_type, is_symmetric, reduce_range)


def _linear_quantize(fp_tensor, quant_type, is_symmetric, reduce_range, scale, zero_point):
    fp_tensor_scaled = fp_tensor / scale
    shifted_tensor = np.round(fp_tensor_scaled).astype(np.int32) + zero_point

    quantized_min, quantized_max = quant_type.qrange(is_symmetric, reduce_range)
    q_tensor = np.clip(shifted_tensor, quantized_min, quantized_max)
    q_tensor = q_tensor.astype(quant_type.np_dtype)

    return q_tensor


def quantize_tensor(
    fp_tensor,
    quant_type,
    *,
    is_symmetric=False,
    reduce_range=False,
    per_channel=False,
    clip_ratio=1.0,
    mse=False,
):
    """Quantizes a tensor using asymmetric quantization.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to quantize.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8). Defaults to QuantType.QInt8.
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.
        reduce_range (bool, optional): Whether to use reduced range for quantization.
            Defaults to False.
        per_channel (bool): Whether to perform per-channel quantization. Defaults to False.
        clip_ratio (float, optional): percentile of clip. Defaults to 1.0
        mse (bool, optional): Whether to use MSE minimization to compute quantization parameters.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The quantized tensor, scale, and zero-point.
    """
    scale, zero_point = get_quantization_params_from_tensor(
        fp_tensor,
        quant_type,
        is_symmetric=is_symmetric,
        reduce_range=reduce_range,
        per_channel=per_channel,
        clip_ratio=clip_ratio,
        mse=mse,
    )
    q_tensor = _linear_quantize(
        fp_tensor, quant_type, is_symmetric, reduce_range, scale, zero_point
    )
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
    return (q_tensor.astype(np.float32) - zero_point.astype(np.float32)) * scale


def fake_quantize_tensor(
    fp_tensor, scale, zero_point, quant_type=QuantType.QInt8, is_symmetric=False, reduce_range=False
):
    """Quantizes a tensor using asymmetric quantization.

    Args:
        fp_tensor (np.ndarray): The floating-point tensor to quantize.
        scale (np.ndarray): The scaling factor.
        zero_point (np.ndarray): The zero point.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8). Defaults to QuantType.QInt8.
        is_symmetric (bool, optional): Whether to use symmetric quantization. Defaults to False.
        reduce_range (bool, optional): Whether to use reduced range for quantization.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The quantized tensor, scale, and zero-point.
    """
    q_tensor = _linear_quantize(
        fp_tensor, quant_type, is_symmetric, reduce_range, scale, zero_point
    )
    return dequantize_tensor(q_tensor, scale, zero_point)


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
        bias,
        QuantType.QInt32,
        is_symmetric=False,
        reduce_range=False,
        scale=bias_scale,
        zero_point=0,
    )
    return qbias, bias_scale, 0

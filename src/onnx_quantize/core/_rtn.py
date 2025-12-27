__all__ = [
    "_preprocess_array",
    "_post_process_array",
    "_compute_min_max",
    "_compute_min_max_mse",
    "_compute_qparams",
    "_compute_qparams_from_array",
    "_quantize_array_from_qparams",
    "_quantize_array",
    "_fake_quantize_array",
    "_dequantize_array",
    "_quantize_bias",
]

import math

import numpy as np

from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QuantizationStrategy


def _preprocess_array(array, strategy, group_size=-1):
    def pad_array(array, group_size, num_groups):
        """Pad array rows so that it can be is divisible by group_size.

        Args:
            array (np.ndarray): weight
            group_size (int): how many elements share one scale/zp
            num_groups (int): the number of groups

        Returns:
            weight: paded weight
        """
        if group_size == -1:
            return array

        org_shape = array.shape
        padded_rows = num_groups * group_size
        pad_len = padded_rows - org_shape[0]

        if pad_len > 0:
            array = np.pad(array, ((0, pad_len), (0, 0)), "constant")

        return array

    assert isinstance(strategy, QuantizationStrategy)

    if strategy == QuantizationStrategy.TENSOR:
        return array

    elif strategy == QuantizationStrategy.CHANNEL:
        return array.T

    elif strategy == QuantizationStrategy.GROUP:
        in_channels = array.shape[0]
        group_size = group_size if group_size != -1 else in_channels
        num_groups = math.ceil(in_channels / group_size)
        array = pad_array(array, group_size, num_groups)

        # (in_channels, out_channels) -> (out_channels x num_groups , group_size)
        array = array.T.reshape((-1, group_size))

        return array


def _post_process_array(preprocessed_array, original_array, strategy, group_size=-1):
    assert isinstance(strategy, QuantizationStrategy)

    if strategy == QuantizationStrategy.TENSOR:
        return preprocessed_array

    elif strategy == QuantizationStrategy.CHANNEL:
        return preprocessed_array.T

    elif strategy == QuantizationStrategy.GROUP:
        return preprocessed_array.reshape(original_array.T.shape).T


def _compute_min_max(array, strategy, group_size=-1, clip_ratio=1.0):
    """Returns the computed scales and zero points for dynamic activation quantization.

    Args:
        array (np.ndarray): The input array to be quantized.
        strategy (QuantizationStrategy): The quantization strategy to use.
        group_size (int): The group size for group quantization.
        clip_ratio (float): The clip ratio to apply to the min and max values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The computed min and max values.
    """
    assert isinstance(strategy, QuantizationStrategy)

    keep_dims = True
    axis = 1
    if strategy == QuantizationStrategy.TENSOR:
        axis = None
        keep_dims = False

    min_val = np.min(array, axis=axis, keepdims=keep_dims) * clip_ratio
    max_val = np.max(array, axis=axis, keepdims=keep_dims) * clip_ratio

    # Include Zero in the range to have a valid zero point
    min_val = np.minimum(min_val, 0)
    max_val = np.maximum(max_val, 0)

    return np.array(min_val), np.array(max_val)


def _compute_min_max_mse(
    array,
    quant_type,
    strategy,
    group_size,
    is_symmetric,
    reduce_range,
    maxshrink=0.20,
    patience=5,
    grid=100.0,
    norm=2.4,
):
    """Computes the optimal min and max values for quantization using MSE minimization.

    This function searches for the best quantization range by iteratively shrinking
    the min/max values and evaluating the mean squared error between the original
    and quantized tensors. It uses early stopping to avoid unnecessary iterations.

    Args:
        array (np.ndarray): The floating-point tensor to be quantized.
        quant_type (QuantType): The quantization data type.
        strategy (QuantizationStrategy): The quantization strategy to use.
        group_size (int): The group size for group quantization.
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
    axis = None if strategy == QuantizationStrategy.TENSOR else 1
    keep_dims = False if strategy == QuantizationStrategy.TENSOR else True
    rmin, rmax = _compute_min_max(array, strategy, group_size, clip_ratio=1.0)

    best_error = np.full_like(rmin, np.finfo(rmin.dtype).max)
    best_min_val = rmin.copy()
    best_max_val = rmax.copy()

    # Early stopping params
    no_improve_count = 0

    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrinked_min_val = p * rmin
        shrinked_max_val = p * rmax

        candidate_scales, candidate_zero_points = _compute_qparams(
            rmin=shrinked_min_val,
            rmax=shrinked_max_val,
            quant_type=quant_type,
            is_symmetric=is_symmetric,
            reduce_range=reduce_range,
        )
        q = _fake_quantize_array(
            array,
            candidate_scales,
            candidate_zero_points,
            quant_type,
            is_symmetric,
            reduce_range,
        )

        q -= array
        q = np.abs(q)
        q = np.power(q, norm)

        err = np.sum(q, axis=axis, keepdims=keep_dims)
        tmp = err < best_error

        # Vector case: boolean mask indexing
        if np.any(tmp):
            best_error[tmp] = err[tmp]
            best_min_val[tmp] = shrinked_min_val[tmp]
            best_max_val[tmp] = shrinked_max_val[tmp]
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            break

    return best_min_val, best_max_val


def _compute_qparams(rmin, rmax, quant_type, is_symmetric, reduce_range):
    """Computes the quantization parameters.

    Args:
        rmin (np.ndarray): The minimum values of the tensor to be quantized.
        rmax (np.ndarray): The maximum values of the tensor to be quantized.
        quant_type (QuantType, optional): The quantization type.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.

    Returns:
        tuple[np.ndarray, np.ndarray]: The quantization scale factor and null zero point.
    """

    def _compute_qparams_asymmetric(rmin, rmax, quant_type):
        qmin, qmax = quant_type.qrange(is_symmetric=False, reduce_range=reduce_range)

        # Compute scale
        scale = (rmax - rmin) / (qmax - qmin)
        scale = np.where(scale < np.finfo(rmax.dtype).tiny, 1, scale)

        # Compute zero point
        zero_point = qmin - (rmin / scale)
        zero_point = np.round(np.clip(zero_point, qmin, qmax))

        return scale.astype(np.float32), np.asarray(zero_point, dtype=quant_type.np_dtype)

    def _compute_qparams_symmetric(rmax, quant_type):
        qmin, qmax = quant_type.qrange(is_symmetric=True, reduce_range=reduce_range)

        # Compute scale
        scale = (2 * rmax) / (qmax - qmin)
        scale = np.where(scale < np.finfo(rmax.dtype).tiny, 1, scale)

        # Compute zero point
        # For symmetric quantization, zero point is always the middle of the range
        # This is because symmetric quantization assumes zero is in the middle of the range
        # Therefore, it is not always equal to 0, but it is the middle of the quantized range
        # e.g., for QInt8, the zero point is 0, but for QUInt8, it is 128
        zero = np.multiply(np.ones(rmax.shape), np.round((qmax + qmin) / 2.0))
        return scale.astype(np.float32), np.asarray(zero, dtype=quant_type.np_dtype)

    if is_symmetric:
        rmax = np.maximum(np.abs(rmin), np.abs(rmax))
        return _compute_qparams_symmetric(rmax, quant_type)
    return _compute_qparams_asymmetric(rmin, rmax, quant_type)


def _compute_qparams_from_array(
    array, quant_type, strategy, group_size, is_symmetric, reduce_range, clip_ratio, mse
):
    """Computes the quantization parameters from a tensor.

    Args:
        array (np.ndarray): The floating-point tensor to be quantized.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8)
        strategy (QuantizationStrategy): The quantization strategy to use.
        group_size (int): The group size for group quantization.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.
        clip_ratio (float): percentile of clip.
        mse (bool): Whether to use MSE minimization to compute quantization parameters.

    Returns:
        tuple[np.ndarray, np.ndarray]: The quantization scale factor and null zero point.
    """
    rmin, rmax = _compute_min_max(array, strategy, group_size, clip_ratio)

    if mse:
        rmin, rmax = _compute_min_max_mse(
            array, quant_type, strategy, group_size, is_symmetric, reduce_range
        )

    return _compute_qparams(rmin, rmax, quant_type, is_symmetric, reduce_range)


def _quantize_array_from_qparams(array, scale, zero_point, quant_type, is_symmetric, reduce_range):
    array_scaled = array / scale
    shifted_tensor = np.round(array_scaled).astype(np.int32) + zero_point

    qmin, qmax = quant_type.qrange(is_symmetric, reduce_range)
    q_array = np.clip(shifted_tensor, qmin, qmax)

    return q_array.astype(quant_type.np_dtype)


def _quantize_array(
    array, quant_type, strategy, group_size, is_symmetric, reduce_range, clip_ratio, mse
):
    """Quantizes a tensor using asymmetric quantization.

    Args:
        array (np.ndarray): The floating-point tensor to quantize.
        quant_type (QuantType, optional): The quantization type, either signed (QInt8)
            or unsigned (QUInt8). Defaults to QuantType.QInt8.
        strategy (QuantizationStrategy): The quantization strategy to use.
        group_size (int): The group size for group quantization.
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


def _fake_quantize_array(array, scale, zero_point, quant_type, is_symmetric, reduce_range):
    """Simulates quantization and dequantization of a tensor.

    Args:
        array (np.ndarray): The floating-point tensor to be quantized and dequantized.
        scale (np.ndarray): The scaling factor.
        zero_point (np.ndarray): The zero point.
        quant_type (QuantType): The quantization data type.
        is_symmetric (bool): Whether to use symmetric quantization.
        reduce_range (bool): Whether to use reduced range for quantization.

    Returns:
        np.ndarray: The fake quantized tensor.
    """
    q_tensor = _quantize_array_from_qparams(
        array, scale, zero_point, quant_type, is_symmetric, reduce_range
    )
    return _dequantize_array(q_tensor, scale, zero_point)


def _dequantize_array(
    q_array, scale, zero_point, *, preprocess=False, strategy=None, group_size=-1
):
    """Dequantizes a tensor.

    Args:
        q_array (np.ndarray): The quantized tensor to dequantize.
        scale (np.ndarray): The scaling factor.
        zero_point (np.ndarray): The zero point.
        preprocess (bool, optional): Whether to preprocess the array before dequantization.
            Defaults to False.
        strategy (QuantizationStrategy, optional): The quantization strategy used
            during preprocessing. Required if preprocess is True.
        group_size (int, optional): The group size for

    Returns:
        np.ndarray: The dequantized tensor
    """
    preprocessed_array = q_array
    if preprocess:
        assert strategy is not None, "strategy must be provided if preprocess is True"
        preprocessed_array = _preprocess_array(q_array, strategy, group_size)

        # Expand scale and zp dims (for tensor strategy, they are scalars,
        # for group, they already have the correct shape)
        if strategy == QuantizationStrategy.CHANNEL:
            scale, zero_point = np.expand_dims(scale, axis=1), np.expand_dims(zero_point, axis=1)

    dequantized_array = (
        preprocessed_array.astype(np.float32) - zero_point.astype(np.float32)
    ) * scale

    if preprocess:
        dequantized_array = _post_process_array(dequantized_array, q_array, strategy, group_size)

    return dequantized_array


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

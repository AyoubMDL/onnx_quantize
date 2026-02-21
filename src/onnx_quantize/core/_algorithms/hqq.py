__all__ = ["_hqq_quantize"]

import numpy as np

from onnx_quantize.core._algorithms.utils import (
    _compute_qparams_from_array,
    _post_process_array,
    _preprocess_array,
)
from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QuantizationStrategy


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _shrink_op(x: np.ndarray, beta: float, lp_norm: float) -> np.ndarray:
    return np.sign(x) * _relu(np.abs(x) - (1.0 / beta) * np.power(np.abs(x) + 1e-8, lp_norm - 1))


def _optimize_zero_point(
    w_f: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
    quant_type: QuantType,
    reduce_range: bool = False,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
    iters: int = 20,
    early_stop: bool = True,
) -> np.ndarray:
    best_error = np.inf
    best_zero_point = zero_point.copy()

    # Hqq uses scale inverted for computation
    scale = 1.0 / scale
    qmin, qmax = quant_type.qrange(is_symmetric=False, reduce_range=reduce_range)

    for _ in range(iters):
        w_q = np.clip(np.round(w_f * scale + zero_point), qmin, qmax)
        w_r = (w_q - zero_point) / scale
        w_e = _shrink_op(w_f - w_r, beta, lp_norm)

        beta *= kappa

        # Compute current error
        current_error = float(np.mean(np.abs(w_f - w_r)))
        if current_error < best_error:
            best_error = current_error
            best_zero_point = zero_point.copy()

        elif early_stop:
            break

        # Update zero point
        zero_point = np.mean(w_q - (w_f - w_e) * scale, axis=1, keepdims=True)

    return best_zero_point


def _hqq_quantize(
    w_f: np.ndarray,
    quant_type: QuantType,
    group_size: int,
    reduce_range: bool = False,
    clip_ratio: float = 1.0,
    mse: bool = False,
    scale_dtype: np.dtype = np.float32,
    zp_dtype: np.dtype = np.float32,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
    iters: int = 20,
    early_stop: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    def quantize(array, scale, zero_point, quant_type, is_symmetric, reduce_range):
        array_scaled = array / scale
        # We don't cast to int32 here
        shifted_tensor = np.round(array_scaled + zero_point)

        qmin, qmax = quant_type.qrange(is_symmetric, reduce_range)
        q_array = np.clip(shifted_tensor, qmin, qmax)

        return q_array.astype(quant_type.np_dtype)

    # In hqq, scale and zero point must have the same dtype
    assert zp_dtype == scale_dtype

    preprocessed_array = _preprocess_array(w_f, QuantizationStrategy.GROUP, group_size)
    scale, zero_point = _compute_qparams_from_array(
        preprocessed_array,
        quant_type,
        QuantizationStrategy.GROUP,
        group_size,
        is_symmetric=False,
        reduce_range=reduce_range,
        clip_ratio=clip_ratio,
        mse=mse,
        scale_dtype=scale_dtype,
        zp_dtype=zp_dtype,
    )

    zero_point = _optimize_zero_point(
        preprocessed_array,
        scale,
        zero_point,
        quant_type,
        reduce_range,
        lp_norm,
        beta,
        kappa,
        iters,
        early_stop,
    )

    w_q = quantize(
        preprocessed_array,
        scale,
        zero_point,
        quant_type,
        is_symmetric=False,
        reduce_range=reduce_range,
    )
    w_q = _post_process_array(w_q, w_f, QuantizationStrategy.GROUP, group_size)

    return w_q, scale, zero_point

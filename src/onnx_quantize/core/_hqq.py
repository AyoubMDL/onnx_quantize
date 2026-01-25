__all__ = ["_hqq_quantize"]

import numpy as np

from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QuantizationStrategy
from onnx_quantize.core._rtn import (
    _preprocess_array,
    _rtn_quantize,
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _shrink_op(x: np.ndarray, beta: float, lp_norm: float) -> np.ndarray:
    return np.sign(x) * _relu(np.abs(x) - (1.0 / beta) * np.power(np.abs(x) + 1e-8, lp_norm - 1))


def _optimize_zero_point(
    w_f: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
    quant_type: QuantType,
    group_size: int,
    reduce_range: bool = False,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
    iters: int = 20,
    early_stop: bool = True,
) -> np.ndarray:
    assert group_size > 0, "Group size must be greater than 0 for HQQ optimization."
    w_f = _preprocess_array(w_f, QuantizationStrategy.GROUP, group_size)

    best_error = 1e4
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

            if early_stop:
                break

        # Update zero point
        zero_point = np.mean(w_q - (w_f - w_e) * scale, axis=1, keepdims=True)

    del w_f, w_q, w_r, w_e
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
    # In hqq, scale and zero point must have the same dtype
    assert zp_dtype == scale_dtype

    w_q, scale, zero_point = _rtn_quantize(
        w_f,
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
        w_f,
        scale,
        zero_point,
        quant_type,
        group_size,
        reduce_range,
        lp_norm,
        beta,
        kappa,
        iters,
        early_stop,
    )

    return w_q, scale, zero_point

__all__ = ["HqqConfig", "_hqq_quantize"]

from typing import TYPE_CHECKING, Literal

import numpy as np

from onnx_quantize.core._algorithms.utils import (
    _compute_qparams_from_array,
    _post_process_array,
    _preprocess_array,
)
from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import (
    AlgorithmConfig,
    QuantizationStrategy,
    register_algorithm_config,
)


if TYPE_CHECKING:
    import onnx_ir as ir

    from onnx_quantize.core._qconfig import QConfig, QWeightArgs


@register_algorithm_config
class HqqConfig(AlgorithmConfig):
    """HqqConfig is the configuration class handling all the HQQ quantization parameters.

    Args:
        lp_norm (float, optional): The Lp norm to use for optimization. Defaults to 0.7.
        beta (float, optional): The beta parameter for the shrinkage operator. Defaults to 10.0.
        kappa (float, optional): The kappa parameter for the optimization. Defaults to 1.01.
        iters (int, optional): The number of iterations for optimization. Defaults to 20.
        early_stop (bool, optional): Whether to use early stopping in optimization.
            Defaults to True.
    """

    algorithm_type: Literal["hqq"] = "hqq"
    lp_norm: float = 0.7
    beta: float = 1e1
    kappa: float = 1.01
    iters: int = 20
    early_stop: bool = True

    @staticmethod
    def _check_hqq_constraints(
        dtype: QuantType, symmetric: bool, strategy: QuantizationStrategy, group_size: int
    ) -> bool:
        if dtype != QuantType.QUInt4:
            raise ValueError(f"HQQ only supports uint4 weight type. Found: {np.dtype}")

        if symmetric:
            raise ValueError("HQQ only supports asymmetric quantization.")

        # TODO: Maybe merge these with is_matmul_nbits_compatible
        if strategy != QuantizationStrategy.GROUP:
            # Because HQQ can only be used with MatMulNBits which expects groups
            raise ValueError(f"HQQ only supports 'group' quantization strategy. Found: {strategy}")

        if group_size != -1 and (group_size < 16 or (group_size & (group_size - 1)) != 0):
            raise ValueError(
                f"HQQ requires group_size to be greater than 16 and a power of 2. "
                f"Found: {group_size}"
            )

    def validate_weight_args(self, weight_args: "QWeightArgs") -> None:
        self._check_hqq_constraints(
            weight_args.dtype,
            weight_args.symmetric,
            weight_args.strategy,
            weight_args.group_size,
        )

        # For HQQ, scale and zp dtypes are the same
        weight_args.zp_dtype = weight_args.scale_dtype

    def quantize_weights(
        self, w: "ir.Value", qconfig: "QConfig", out: "ir.Value | None" = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _hqq_quantize(
            w.const_value.numpy(),
            quant_type=qconfig.weights.dtype,
            group_size=qconfig.weights.group_size,
            reduce_range=qconfig.weights.reduce_range,
            clip_ratio=qconfig.weights.clip_ratio,
            mse=qconfig.weights.mse,
            scale_dtype=qconfig.weights.scale_dtype,
            zp_dtype=qconfig.weights.zp_dtype,
            lp_norm=self.lp_norm,
            beta=self.beta,
            kappa=self.kappa,
            iters=self.iters,
            early_stop=self.early_stop,
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

__all__ = ["_gptq_quantize"]

import copy
import math

import numpy as np

from onnx_quantize.core._dtypes import QuantType
from onnx_quantize.core._qconfig import QuantizationStrategy
from onnx_quantize.core._rtn import (
    _compute_qparams_from_array,
    _dequantize_array,
    _quantize_array_from_qparams,
)


def _gptq(
    W: np.array,
    H: np.array,
    quant_type=QuantType.QUInt8,
    strategy=QuantizationStrategy.CHANNEL,
    group_size=32,
    is_symmetric=False,
    reduce_range=False,
    clip_ratio=1.0,
    block_size=128,
    percdamp=0.01,
    actorder=False,
    mse=False,
):
    # Create writable copies of the input arrays
    W = W.copy()
    H = H.copy()

    shape = W.shape
    # TODO: change this
    scale, zp = _compute_qparams_from_array(
        W.T,
        quant_type=quant_type,
        strategy=strategy,
        group_size=-1,
        is_symmetric=is_symmetric,
        clip_ratio=clip_ratio,
        reduce_range=reduce_range,
        mse=mse,
    )
    scale, zp = np.squeeze(scale), np.squeeze(zp)

    # mask dead hessian values
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[dead, :] = 0  # such channel makes no contribution to quantization computation

    # rearrange considering the diag's value
    if actorder:
        perm = np.argsort(np.diag(H))[::-1]
        W = W[perm, :]
        H = H[perm, :][:, perm]

    losses = np.zeros_like(W)
    Q = np.zeros_like(W)
    Q_int = np.zeros_like(W)

    # compute inverse hessian
    damp = percdamp * np.mean(np.diag(H))
    diag = np.arange(shape[0])

    H[diag, diag] += damp  # add a average value of
    H = np.linalg.cholesky(H)
    H = np.linalg.inv(H)
    H = np.linalg.cholesky(H.T @ H).T
    Hinv = H

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, shape[0], block_size):
        i2 = min(i1 + block_size, shape[0])
        count = i2 - i1

        W1 = copy.deepcopy(W[i1:i2, :])
        Q1 = np.zeros_like(W1)
        Q1_int = np.zeros_like(W1)
        Err1 = np.zeros_like(W1)
        Losses1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):  # within a block, channel wise
            w = W1[i, :]
            d = Hinv1[i, i]

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    scale, zp = _compute_qparams_from_array(
                        W[(i1 + i) : (i1 + i + group_size), :].T,
                        quant_type=quant_type,
                        strategy=strategy,
                        group_size=-1,
                        is_symmetric=is_symmetric,
                        reduce_range=reduce_range,
                        clip_ratio=clip_ratio,
                        mse=mse,
                    )
                    scale, zp = np.squeeze(scale), np.squeeze(zp)

            q_int = _quantize_array_from_qparams(
                w, scale, zp, quant_type, is_symmetric, reduce_range
            ).flatten()
            q = _dequantize_array(q_int, scale, zp)

            # propagate column error
            Q1[i, :] = q
            Q1_int[i, :] = q_int

            Losses1[i, :] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[i:, :] -= np.matmul(
                np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0)
            )
            Err1[i, :] = err1

        # propagate block error
        Q[i1:i2, :] = Q1
        Q_int[i1:i2, :] = Q1_int
        losses[i1:i2, :] = Losses1 / 2

        W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

    if actorder:
        invperm = np.argsort(perm)
        Q = Q[invperm, :]
        Q_int = Q_int[invperm, :]

    Q = np.reshape(Q, W.shape)
    Q_int = Q_int.reshape(W.shape).astype(quant_type.np_dtype)
    scale = scale.astype(np.float32)
    zp = zp.astype(Q_int.dtype)
    del W
    del Q
    return Q_int, scale, zp


def _accumulate_hessian(inp, H, num_samples):
    num_added = inp.shape[0]

    # This only for Linear layers
    # Check https://github.com/vllm-project/llm-compressor/blob/99e231e16d7ef45e2fab67c4c77178900eb00f33/src/llmcompressor/modifiers/quantization/gptq/gptq_quantize.py#L50
    # for Conv2d
    inp = np.reshape(inp, (-1, inp.shape[-1]))

    H *= num_samples / (num_samples + num_added)
    num_samples += num_added

    inp = math.sqrt(2 / num_samples) * inp.astype(np.float32)
    H += np.matmul(np.transpose(inp), inp)

    return H, num_samples


def _gptq_quantize(
    weights,
    inputs,
    quant_type=QuantType.QInt8,
    strategy=QuantizationStrategy.CHANNEL,
    group_size=32,
    is_symmetric=False,
    reduce_range=False,
    clip_ratio=1.0,
    block_size=128,
    percdamp=0.01,
    actorder=False,
    mse=False,
):
    """Quant the weight with GPTQ method.

    Args:
        weights (np.array): weight.
        inputs (np.array): input activations.
        quant_type (QuantType, optional): quantization type. Default is QuantType.QInt8.
        strategy (QuantizationStrategy, optional): quantization strategy. Default is CHANNEL.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        is_symmetric (bool, optional): sym or asym. Defaults to False.
        reduce_range (bool, optional): Whether to use reduced range for quantization.
            Defaults to False.
        clip_ratio (float, optional): percentile of clip. Defaults to 1.0
        block_size (int, optional): block_size to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        per_channel (bool, optional): whether quantize weight per-channel.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): quantized weight, scale, zero point.
    """
    # Compute Hessian
    num_samples = 0
    H = np.zeros((weights.shape[0], weights.shape[0]), dtype=np.float32)
    H, num_samples = _accumulate_hessian(inputs, H, num_samples)

    w_q, w_scale, w_zero_point = _gptq(
        weights,
        H,
        quant_type=quant_type,
        strategy=strategy,
        is_symmetric=is_symmetric,
        reduce_range=reduce_range,
        clip_ratio=clip_ratio,
        group_size=group_size,
        block_size=block_size,
        percdamp=percdamp,
        actorder=actorder,
        mse=mse,
    )

    return w_q, w_scale, w_zero_point

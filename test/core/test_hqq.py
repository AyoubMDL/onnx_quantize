import numpy as np
import pytest

from onnx_quantize import QuantizationStrategy, QuantType
from onnx_quantize.core._hqq import _hqq_quantize
from onnx_quantize.core._rtn import _dequantize_array


@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("mse", [True, False])
@pytest.mark.parametrize("early_stop", [True, False])
def test_hqq_quantize(rng, group_size, mse, early_stop):
    W_f = rng.standard_normal((32, 64)).astype(np.float32)

    w_q, scale, zero_point = _hqq_quantize(
        W_f,
        quant_type=QuantType.QUInt4,
        group_size=group_size,
        reduce_range=False,
        clip_ratio=1.0,
        mse=mse,
        early_stop=early_stop,
    )

    assert zero_point.dtype == scale.dtype

    w_r = _dequantize_array(
        w_q,
        scale,
        zero_point,
        preprocess=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
    )

    np.testing.assert_allclose(W_f, w_r, atol=5e-1)


@pytest.mark.parametrize(
    "lp_norm, beta, kappa, iters",
    [
        (0.7, 10.0, 1.01, 20),
        (0.5, 5.0, 1.05, 10),
        (1.0, 15.0, 1.02, 15),
    ],
)
def test_hqq_quantize_custom_params(rng, lp_norm, beta, kappa, iters):
    W_f = rng.standard_normal((32, 64)).astype(np.float32)

    w_q, scale, zero_point = _hqq_quantize(
        W_f,
        quant_type=QuantType.QUInt4,
        group_size=32,
        lp_norm=lp_norm,
        beta=beta,
        kappa=kappa,
        iters=iters,
    )

    assert zero_point.dtype == scale.dtype

    w_r = _dequantize_array(
        w_q,
        scale,
        zero_point,
        preprocess=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    )

    np.testing.assert_allclose(W_f, w_r, atol=5e-1)


@pytest.mark.parametrize("reduce_range", [True, False])
def test_hqq_quantize_reduce_range(rng, reduce_range):
    W_f = rng.standard_normal((32, 64)).astype(np.float32)

    w_q, _, _ = _hqq_quantize(
        W_f,
        quant_type=QuantType.QUInt4,
        group_size=32,
        reduce_range=reduce_range,
    )

    # Check quantized values are within expected range
    qmin, qmax = QuantType.QUInt4.qrange(is_symmetric=False, reduce_range=reduce_range)
    assert np.all(w_q >= qmin)
    assert np.all(w_q <= qmax)


@pytest.mark.parametrize("clip_ratio", [0.95, 1.0])
def test_hqq_quantize_clip_ratio(rng, clip_ratio):
    W_f = rng.standard_normal((32, 64)).astype(np.float32)

    w_q, scale, zero_point = _hqq_quantize(
        W_f,
        quant_type=QuantType.QUInt4,
        group_size=32,
        clip_ratio=clip_ratio,
    )

    assert zero_point.dtype == scale.dtype

    w_r = _dequantize_array(
        w_q,
        scale,
        zero_point,
        preprocess=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    )

    np.testing.assert_allclose(W_f, w_r, atol=5e-1)


def test_hqq_quantize_dtype_consistency(rng):
    W_f = rng.standard_normal((32, 64)).astype(np.float32)

    w_q, scale, zero_point = _hqq_quantize(
        W_f,
        quant_type=QuantType.QUInt4,
        group_size=32,
        scale_dtype=np.float32,
        zp_dtype=np.float32,
    )

    assert scale.dtype == np.float32
    assert zero_point.dtype == np.float32


def test_hqq_quantize_assertion_error():
    W_f = np.random.randn(32, 64).astype(np.float32)

    with pytest.raises(AssertionError):
        _hqq_quantize(
            W_f,
            quant_type=QuantType.QUInt4,
            group_size=32,
            scale_dtype=np.float32,
            zp_dtype=np.float16,
        )

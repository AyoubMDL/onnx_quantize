import numpy as np

from onnx_quantize.core._qconfig import QConfig, QWeightArgs
from onnx_quantize.qrules._common import _prepare_for_matmul_nbits


def test_prepare_for_matmul_nbits_odd_num_blocks():
    group_size, num_blocks, out_channels = 16, 5, 4
    in_channels = group_size * num_blocks
    qconfig = QConfig(
        weights=QWeightArgs(dtype="uint4", strategy="group", group_size=group_size),
    )

    rng = np.random.default_rng(0)
    w_q = rng.integers(0, 16, size=(in_channels, out_channels), dtype=np.uint8)
    w_scale = rng.random(size=(out_channels * num_blocks,)).astype(np.float32)
    w_zp = rng.integers(0, 16, size=(out_channels * num_blocks, 1), dtype=np.uint8)

    _, _, packed_zp = _prepare_for_matmul_nbits(w_q, w_scale, w_zp, qconfig)

    assert packed_zp.shape == (out_channels, (num_blocks + 1) // 2)

    # Unpack and check each row's first num_blocks nibbles match the input zps for that row.
    lower = packed_zp & 0x0F
    upper = (packed_zp >> 4) & 0x0F
    nibbles = np.empty((out_channels, packed_zp.shape[1] * 2), dtype=np.uint8)
    nibbles[:, ::2] = lower
    nibbles[:, 1::2] = upper
    np.testing.assert_array_equal(
        nibbles[:, :num_blocks], w_zp.reshape(out_channels, num_blocks)
    )

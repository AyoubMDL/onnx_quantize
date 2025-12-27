import pytest

from onnx_quantize import QuantType


@pytest.mark.parametrize(
    "quant_type, symmetric, reduce_range, expected",
    [
        # Int4
        (QuantType.QInt4, False, False, (-8, 7)),
        (QuantType.QInt4, True, False, (-7, 7)),
        (QuantType.QInt4, True, True, (-4, 3)),
        # UInt4
        (QuantType.QUInt4, False, False, (0, 15)),
        (QuantType.QUInt4, True, False, (0, 15)),
        (QuantType.QUInt4, True, True, (0, 7)),
        # Int8
        (QuantType.QInt8, False, False, (-128, 127)),
        (QuantType.QInt8, True, False, (-127, 127)),
        (QuantType.QInt8, True, True, (-64, 64)),
        # UInt8
        (QuantType.QUInt8, False, False, (0, 255)),
        (QuantType.QUInt8, True, False, (0, 255)),
        (QuantType.QUInt8, True, True, (0, 127)),
        # Int32
        (QuantType.QInt32, False, False, (-(2**31), 2**31 - 1)),
        (QuantType.QInt32, True, False, (-(2**31 - 1), 2**31 - 1)),
        (QuantType.QInt32, True, True, (-(2**30), 2**30)),
        # UInt32
        (QuantType.QUInt32, False, False, (0, 2**32 - 1)),
        (QuantType.QUInt32, True, False, (0, 2**32 - 1)),
        (QuantType.QUInt32, True, True, (0, 2**31 - 1)),
    ],
)
def test_quant_type(quant_type, symmetric, reduce_range, expected):
    result = quant_type.qrange(symmetric, reduce_range)
    assert result == expected

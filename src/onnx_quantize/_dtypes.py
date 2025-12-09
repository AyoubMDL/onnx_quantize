import enum

import onnx_ir as ir


_DTYPE_RANGES = {
    ir.DataType.UINT4: (0, 15),
    ir.DataType.INT4: (-8, 7),
    ir.DataType.UINT8: (0, 255),
    ir.DataType.INT8: (-128, 127),
    ir.DataType.UINT32: (0, 2**32 - 1),
    ir.DataType.INT32: (-(2**31), 2**31 - 1),
}

_SYMMETRIC_RANGES = {
    ir.DataType.INT4: (-7, 7),
    ir.DataType.INT8: (-127, 127),
    ir.DataType.INT32: (-(2**31 - 1), 2**31 - 1),
}

_REDUCED_RANGES = {
    ir.DataType.UINT4: (0, 7),
    ir.DataType.INT4: (-4, 3),
    ir.DataType.UINT8: (0, 127),
    ir.DataType.INT8: (-64, 64),
    ir.DataType.UINT32: (0, 2**31 - 1),
    ir.DataType.INT32: (-(2**30), 2**30),
}


class QuantType(enum.Enum):
    """Enumeration of quantization types."""

    QInt4 = ir.DataType.INT4
    QUInt4 = ir.DataType.UINT4
    QInt8 = ir.DataType.INT8
    QUInt8 = ir.DataType.UINT8
    QInt32 = ir.DataType.INT32
    QUInt32 = ir.DataType.UINT32

    @property
    def np_dtype(self):
        return self.value.numpy()

    def qrange(self, is_symmetric, reduce_range=False):
        dtype = self.value
        if reduce_range:
            qrange = _REDUCED_RANGES.get(dtype)
        elif is_symmetric and dtype in _SYMMETRIC_RANGES:
            qrange = _SYMMETRIC_RANGES[dtype]
        else:
            qrange = _DTYPE_RANGES.get(dtype)

        return qrange

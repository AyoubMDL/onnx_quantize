__all__ = ["pack", "unpack"]

import numpy as np

from onnx_quantize.core._dtypes import QuantType


def _pack_4bitx2(array):
    # Taken from onnx_ir/_type_casting.py
    """Convert a numpy array to flatten, packed int4/uint4.

    Elements must be in the correct range.
    """
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]  # type: ignore[return-type]


def _unpack_4bitx2(data, dims):
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8."""
    # Taken from onnx_ir/_type_casting.py
    assert data.dtype == np.uint8, "Input data must be of type uint8"
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def _pack_uint4(array):
    """Pack unsigned uint4 values (0 to 15) into uint8."""
    return _pack_4bitx2(array.astype(np.uint8))


def _unpack_uint4(data, dims):
    """Unpack to unsigned uint4 values represented as uint8."""
    return _unpack_4bitx2(data, dims)


def _pack_int4(array):
    """Pack signed int4 values (-8 to 7) into uint8."""
    # Convert signed int4 to unsigned representation (0-15)
    array_unsigned = array.copy()
    array_unsigned = np.where(array_unsigned < 0, array_unsigned + 16, array_unsigned)
    return _pack_4bitx2(array_unsigned.astype(np.uint8))


def _unpack_int4(data, dims):
    """Unpack to signed int4 values represented as int8."""
    result = _unpack_4bitx2(data, dims)
    # Convert from unsigned 4-bit to signed: if bit 3 is set, it's negative
    result = result.astype(np.int8)
    result = np.where(result > 7, result - 16, result)
    return result


def _pack_4bit(array, quant_type):
    """Pack signed int4 values (-8 to 7) into uint8."""
    assert quant_type in (QuantType.QInt4, QuantType.QUInt4), "quant_type must be QInt4 or QUInt4"
    if quant_type == QuantType.QInt4:
        return _pack_int4(array)
    return _pack_uint4(array)


def _unpack_4bit(array, dims, quant_type):
    """Unpack signed int4 values (-8 to 7) from uint8."""
    assert quant_type in (QuantType.QInt4, QuantType.QUInt4), "quant_type must be QInt4 or QUInt4"
    if quant_type == QuantType.QInt4:
        return _unpack_int4(array, dims)
    return _unpack_uint4(array, dims)


def pack(array, quant_type):
    """Pack array based on quantization type.

    Args:
        array (np.ndarray): Input array to be packed.
        quant_type (QuantType): Quantization type.

    Returns:
        np.ndarray: Packed array.
    """
    if quant_type in (QuantType.QInt4, QuantType.QUInt4):
        return _pack_4bit(array, quant_type)
    # Placeholder for other quantization types (int2, uint2, etc.)
    return array.astype(quant_type.np_dtype)


def unpack(array, dims, quant_type):
    """Unpack array based on quantization type.

    Args:
        array (np.ndarray): Input packed array.
        dims (tuple): Desired output dimensions.
        quant_type (QuantType): Quantization type.

    Returns:
        np.ndarray: Unpacked array.
    """
    if quant_type in (QuantType.QInt4, QuantType.QUInt4):
        return _unpack_4bit(array, dims, quant_type)
    # Placeholder for other quantization types (int2, uint2, etc.)
    return array.astype(quant_type.np_dtype)

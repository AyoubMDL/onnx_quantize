import numpy as np
import pytest

from onnx_quantize import QuantType
from onnx_quantize._pack import pack, unpack


class TestPackUnpackInt4:
    """Test pack and unpack functions for QInt4."""

    @pytest.mark.parametrize(
        "array, expected_packed",
        [
            # Simple positive values
            (np.array([3, 7], dtype=np.int8), np.array([115], dtype=np.uint8)),
            # Mixed positive and negative
            (
                np.array([-5, 3, 4, 7, 0, 3, 7, -2], dtype=np.int8),
                np.array([59, 116, 48, 231], dtype=np.uint8),
            ),
            # Edge cases: min and max values
            (np.array([-8, 7], dtype=np.int8), np.array([120], dtype=np.uint8)),
            # All zeros
            (np.array([0, 0, 0, 0], dtype=np.int8), np.array([0, 0], dtype=np.uint8)),
            # All negative
            (np.array([-1, -2, -3, -4], dtype=np.int8), np.array([239, 205], dtype=np.uint8)),
            # Odd number of elements (should pad)
            (np.array([1, 2, 3], dtype=np.int8), np.array([33, 3], dtype=np.uint8)),
        ],
    )
    def test_pack_int4(self, array, expected_packed):
        result = pack(array, QuantType.QInt4)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected_packed)

        # unpack
        unpacked = unpack(result, array.shape, QuantType.QInt4)
        assert unpacked.dtype == np.int8
        np.testing.assert_array_equal(unpacked, array)

    def test_pack_int4_2d_array(self):
        """Test packing 2D array."""
        array = np.array([[1, 2], [3, 4]], dtype=np.int8)
        packed = pack(array, QuantType.QInt4)
        unpacked = unpack(packed, array.shape, QuantType.QInt4)
        np.testing.assert_array_equal(unpacked, array)

    def test_pack_int4_3d_array(self):
        """Test packing 3D array."""
        array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, -1]]], dtype=np.int8)
        packed = pack(array, QuantType.QInt4)
        unpacked = unpack(packed, array.shape, QuantType.QInt4)
        np.testing.assert_array_equal(unpacked, array)


class TestPackUnpackUInt4:
    """Test pack and unpack functions for QUInt4."""

    @pytest.mark.parametrize(
        "array, expected_packed",
        [
            # Simple values
            (np.array([3, 7], dtype=np.uint8), np.array([115], dtype=np.uint8)),
            # Various values
            (
                np.array([11, 3, 4, 7, 0, 3, 7, 14], dtype=np.uint8),
                np.array([59, 116, 48, 231], dtype=np.uint8),
            ),
            # Edge cases: min and max values
            (np.array([0, 15], dtype=np.uint8), np.array([240], dtype=np.uint8)),
            # All zeros
            (np.array([0, 0, 0, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8)),
            # All max values
            (np.array([15, 15, 15, 15], dtype=np.uint8), np.array([255, 255], dtype=np.uint8)),
            # Odd number of elements
            (np.array([1, 2, 3], dtype=np.uint8), np.array([33, 3], dtype=np.uint8)),
        ],
    )
    def test_pack_uint4(self, array, expected_packed):
        result = pack(array, QuantType.QUInt4)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected_packed)

        # unpack
        unpacked = unpack(result, array.shape, QuantType.QUInt4)
        assert unpacked.dtype == np.uint8
        np.testing.assert_array_equal(unpacked, array)

    def test_pack_uint4_2d_array(self):
        """Test packing 2D array."""
        array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        packed = pack(array, QuantType.QUInt4)
        unpacked = unpack(packed, array.shape, QuantType.QUInt4)
        np.testing.assert_array_equal(unpacked, array)

    def test_pack_uint4_3d_array(self):
        """Test packing 3D array."""
        array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
        packed = pack(array, QuantType.QUInt4)
        unpacked = unpack(packed, array.shape, QuantType.QUInt4)
        np.testing.assert_array_equal(unpacked, array)


class TestPackUnpackOtherDTypes:
    """Test pack and unpack for non-4-bit quantization types."""

    @pytest.mark.parametrize(
        "quant_type, dtype",
        [
            (QuantType.QInt8, np.int8),
            (QuantType.QUInt8, np.uint8),
            (QuantType.QInt32, np.int32),
            (QuantType.QUInt32, np.uint32),
        ],
    )
    def test_pack_unpack_passthrough(self, quant_type, dtype):
        """Test that non-4-bit types pass through without packing."""
        array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        packed = pack(array, quant_type)
        # Should be same as input (just dtype conversion)
        np.testing.assert_array_equal(packed, array.astype(quant_type.np_dtype))

        # Unpack should also work
        unpacked = unpack(packed, array.shape, quant_type)
        np.testing.assert_array_equal(unpacked, array.astype(quant_type.np_dtype))


class TestPackUnpackEdgeCases:
    def test_pack_single_element(self):
        """Test packing a single element."""
        array = np.array([5], dtype=np.int8)
        packed = pack(array, QuantType.QInt4)
        unpacked = unpack(packed, array.shape, QuantType.QInt4)
        np.testing.assert_array_equal(unpacked, array)

    def test_pack_large_array(self):
        """Test packing a large array."""
        array = np.random.randint(-8, 8, size=1000, dtype=np.int8)
        packed = pack(array, QuantType.QInt4)
        unpacked = unpack(packed, array.shape, QuantType.QInt4)
        np.testing.assert_array_equal(unpacked, array)

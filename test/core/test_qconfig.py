import numpy as np
import pytest

from onnx_quantize import GPTQConfig, QConfig, QuantizationStrategy, QuantType


def test_quantization_strategy_enum_values():
    """Test that QuantizationStrategy enum has expected values."""
    assert QuantizationStrategy.TENSOR == "tensor"
    assert QuantizationStrategy.CHANNEL == "channel"
    assert QuantizationStrategy.GROUP == "group"


def test_quantization_strategy_from_string():
    """Test creating QuantizationStrategy from string."""
    assert QuantizationStrategy("tensor") == QuantizationStrategy.TENSOR
    assert QuantizationStrategy("channel") == QuantizationStrategy.CHANNEL
    assert QuantizationStrategy("group") == QuantizationStrategy.GROUP


def test_gptq_config_defaults():
    """Test GPTQConfig with default values."""
    config = GPTQConfig()
    assert config.block_size == 128
    assert config.percdamp == 0.01
    assert config.group_size == -1
    assert config.actorder is False


def test_gptq_config_custom_values():
    """Test GPTQConfig with custom values."""
    config = GPTQConfig(
        block_size=256,
        percdamp=0.05,
        group_size=128,
        actorder=True,
    )
    assert config.block_size == 256
    assert config.percdamp == 0.05
    assert config.group_size == 128
    assert config.actorder is True


def test_gptq_config_partial_override():
    """Test GPTQConfig with partial parameter override."""
    config = GPTQConfig(block_size=64, actorder=True)
    assert config.block_size == 64
    assert config.percdamp == 0.01
    assert config.group_size == -1
    assert config.actorder is True


def test_qconfig_defaults():
    """Test QConfig with all default values."""
    config = QConfig()
    assert config.is_static is True
    assert config.weights_only is False
    assert config.clip_ratio == 1.0
    assert config.reduce_range is False
    assert config.group_size is None
    assert config.strategy == QuantizationStrategy.TENSOR
    assert config.mse is False
    assert config.calibration_data is None
    assert config.activations_dtype == QuantType.QUInt8
    assert config.activations_symmetric is False
    assert config.weights_dtype == QuantType.QInt8
    assert config.weights_symmetric is True
    assert config.algorithm is None


def test_qconfig_custom_basic_values():
    """Test QConfig with custom basic values."""
    config = QConfig(
        is_static=False,
        weights_only=True,
        clip_ratio=0.95,
        reduce_range=True,
        mse=True,
        activations_symmetric=True,
        weights_symmetric=False,
    )
    assert config.is_static is False
    assert config.weights_only is True
    assert config.clip_ratio == 0.95
    assert config.reduce_range is True
    assert config.mse is True
    assert config.activations_symmetric is True
    assert config.weights_symmetric is False


def test_qconfig_with_calibration_data():
    """Test QConfig with calibration data."""
    calib_data = np.random.randn(10, 5).astype(np.float32)
    config = QConfig(calibration_data=calib_data)
    assert config.calibration_data is not None
    assert np.array_equal(config.calibration_data, calib_data)


def test_qconfig_weights_dtype_from_quant_type():
    """Test setting weights_dtype using QuantType enum."""
    for dtype in [
        QuantType.QInt4,
        QuantType.QInt8,
        QuantType.QInt32,
        QuantType.QUInt4,
        QuantType.QUInt8,
        QuantType.QUInt32,
    ]:
        config = QConfig(weights_dtype=dtype, weights_only=True)
        assert config.weights_dtype == dtype


def test_qconfig_weights_dtype_from_string():
    """Test setting weights_dtype using string."""
    config = QConfig(weights_dtype="int4", weights_only=True)
    assert config.weights_dtype == QuantType.QInt4

    config = QConfig(weights_dtype="uint8")
    assert config.weights_dtype == QuantType.QUInt8

    config = QConfig(weights_dtype="int32")
    assert config.weights_dtype == QuantType.QInt32


def test_qconfig_activations_dtype_from_quant_type():
    """Test setting activations_dtype using QuantType enum."""
    for dtype in [
        QuantType.QInt8,
        QuantType.QInt32,
        QuantType.QUInt8,
        QuantType.QUInt32,
    ]:
        config = QConfig(activations_dtype=dtype)
        assert config.activations_dtype == dtype


def test_qconfig_activations_dtype_from_string():
    """Test setting activations_dtype using string."""
    config = QConfig(activations_dtype="int8")
    assert config.activations_dtype == QuantType.QInt8

    config = QConfig(activations_dtype="uint32")
    assert config.activations_dtype == QuantType.QUInt32


@pytest.mark.parametrize("insupported_dtype", ["int4", "uint4"])
def test_qconfig_activations_dtype_unsupported(insupported_dtype):
    with pytest.raises(ValueError, match="4-bit quantization is not supported for activations"):
        QConfig(activations_dtype=insupported_dtype)


@pytest.mark.parametrize("insupported_dtype", ["int4", "uint4"])
def test_qconfig_weights_dtype_unsupported(insupported_dtype):
    with pytest.raises(ValueError, match="4-bit quantization is only supported for weights_only"):
        QConfig(weights_dtype=insupported_dtype, weights_only=False)


def test_qconfig_strategy_tensor_inferred_from_none_group_size():
    """Test that strategy is inferred as TENSOR when group_size is None."""
    config = QConfig(group_size=None)
    assert config.strategy == QuantizationStrategy.TENSOR
    assert config.group_size is None


def test_qconfig_strategy_channel_inferred_from_negative_one_group_size():
    """Test that strategy is inferred as CHANNEL when group_size is -1."""
    config = QConfig(group_size=-1)
    assert config.strategy == QuantizationStrategy.CHANNEL
    assert config.group_size == -1


def test_qconfig_strategy_group_inferred_from_positive_group_size():
    """Test that strategy is inferred as GROUP when group_size is positive."""
    config = QConfig(group_size=128, weights_only=True)
    assert config.strategy == QuantizationStrategy.GROUP
    assert config.group_size == 128


def test_qconfig_strategy_explicit_tensor():
    """Test explicitly setting strategy to TENSOR."""
    config = QConfig(strategy=QuantizationStrategy.TENSOR)
    assert config.strategy == QuantizationStrategy.TENSOR


def test_qconfig_strategy_explicit_channel():
    """Test explicitly setting strategy to CHANNEL."""
    config = QConfig(strategy=QuantizationStrategy.CHANNEL)
    assert config.strategy == QuantizationStrategy.CHANNEL


def test_qconfig_strategy_explicit_group():
    """Test explicitly setting strategy to GROUP."""
    config = QConfig(strategy=QuantizationStrategy.GROUP, group_size=64, weights_only=True)
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_strategy_from_string():
    """Test setting strategy from string."""
    config = QConfig(strategy="tensor")
    assert config.strategy == QuantizationStrategy.TENSOR

    config = QConfig(strategy="channel")
    assert config.strategy == QuantizationStrategy.CHANNEL

    config = QConfig(strategy="group", group_size=32, weights_only=True)
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_strategy_from_string_case_insensitive():
    """Test setting strategy from string is case insensitive."""
    config = QConfig(strategy="TENSOR")
    assert config.strategy == QuantizationStrategy.TENSOR

    config = QConfig(strategy="Channel")
    assert config.strategy == QuantizationStrategy.CHANNEL

    config = QConfig(strategy="GROUP", group_size=16, weights_only=True)
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_group_size_various_positive_values():
    """Test various positive group_size values."""
    for group_size in [1, 16, 32, 64, 128, 256, 512]:
        config = QConfig(group_size=group_size, weights_only=True)
        assert config.group_size == group_size
        assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_group_size_invalid_negative():
    """Test that group_size < -1 raises ValueError."""
    with pytest.raises(ValueError, match="Invalid group size"):
        QConfig(group_size=-2)

    with pytest.raises(ValueError, match="Invalid group size"):
        QConfig(group_size=-100)


def test_qconfig_clip_ratio_valid_values():
    """Test valid clip_ratio values."""
    for ratio in [0.001, 0.1, 0.5, 0.9, 0.99, 1.0]:
        config = QConfig(clip_ratio=ratio)
        assert config.clip_ratio == ratio


def test_qconfig_clip_ratio_invalid_zero():
    """Test that clip_ratio = 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="clip_ratio must be in"):
        QConfig(clip_ratio=0.0)

    with pytest.raises(ValueError, match="clip_ratio must be in"):
        QConfig(clip_ratio=-0.1)

    with pytest.raises(ValueError, match="clip_ratio must be in"):
        QConfig(clip_ratio=1.1)

    with pytest.raises(ValueError, match="clip_ratio must be in"):
        QConfig(clip_ratio=2.0)


def test_qconfig_with_gptq_algorithm():
    """Test QConfig with GPTQ algorithm."""
    gptq_config = GPTQConfig(block_size=256, percdamp=0.02)
    config = QConfig(algorithm=gptq_config)
    assert config.algorithm is not None
    assert config.algorithm.block_size == 256
    assert config.algorithm.percdamp == 0.02


def test_qconfig_gptq_with_tensor_strategy():
    """Test that GPTQ is allowed with TENSOR strategy."""
    gptq_config = GPTQConfig()
    config = QConfig(algorithm=gptq_config, strategy=QuantizationStrategy.TENSOR)
    assert config.strategy == QuantizationStrategy.TENSOR
    assert config.algorithm is not None


def test_qconfig_gptq_with_channel_strategy():
    """Test that GPTQ is allowed with CHANNEL strategy."""
    gptq_config = GPTQConfig()
    config = QConfig(algorithm=gptq_config, strategy=QuantizationStrategy.CHANNEL)
    assert config.strategy == QuantizationStrategy.CHANNEL
    assert config.algorithm is not None


def test_qconfig_gptq_with_group_strategy_raises_error():
    """Test that GPTQ with GROUP strategy raises ValueError."""
    gptq_config = GPTQConfig()
    with pytest.raises(ValueError, match="GPTQ algorithm only supports"):
        QConfig(
            algorithm=gptq_config,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            weights_only=True,
        )


def test_qconfig_gptq_inferred_group_strategy_raises_error():
    """Test that GPTQ with inferred GROUP strategy raises ValueError."""
    gptq_config = GPTQConfig()
    with pytest.raises(ValueError, match="GPTQ algorithm only supports"):
        QConfig(algorithm=gptq_config, group_size=128, weights_only=True)


def test_qconfig_group_quantization_requires_weights_only():
    """Test that GROUP strategy requires weights_only=True."""
    with pytest.raises(ValueError, match="Group quantization is only supported for weights_only"):
        QConfig(strategy=QuantizationStrategy.GROUP, group_size=128, weights_only=False)


def test_qconfig_group_quantization_inferred_requires_weights_only():
    """Test that inferred GROUP strategy requires weights_only=True."""
    with pytest.raises(ValueError, match="Group quantization is only supported for weights_only"):
        QConfig(group_size=128, weights_only=False)


def test_qconfig_group_strategy_requires_positive_group_size():
    """Test that GROUP strategy requires positive group_size."""
    with pytest.raises(ValueError, match="requires group_size to be set to a positive value"):
        QConfig(
            strategy=QuantizationStrategy.GROUP,
            group_size=None,
            weights_only=True,
        )

    with pytest.raises(ValueError, match="requires group_size to be set to a positive value"):
        QConfig(
            strategy=QuantizationStrategy.GROUP,
            group_size=-1,
            weights_only=True,
        )


def test_qconfig_positive_group_size_requires_group_strategy():
    """Test that positive group_size requires GROUP strategy."""
    with pytest.raises(ValueError, match="group_size requires strategy to be set to 'group'"):
        QConfig(group_size=128, strategy=QuantizationStrategy.TENSOR, weights_only=True)

    with pytest.raises(ValueError, match="group_size requires strategy to be set to 'group'"):
        QConfig(group_size=128, strategy=QuantizationStrategy.CHANNEL, weights_only=True)


def test_qconfig_weights_only_with_various_strategies():
    """Test weights_only quantization with different strategies."""
    # Tensor strategy
    config = QConfig(weights_only=True, strategy=QuantizationStrategy.TENSOR)
    assert config.weights_only is True
    assert config.strategy == QuantizationStrategy.TENSOR

    # Channel strategy
    config = QConfig(weights_only=True, strategy=QuantizationStrategy.CHANNEL)
    assert config.weights_only is True
    assert config.strategy == QuantizationStrategy.CHANNEL

    # Group strategy
    config = QConfig(weights_only=True, strategy=QuantizationStrategy.GROUP, group_size=64)
    assert config.weights_only is True
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_static_quantization_full_config():
    """Test a complete static quantization configuration."""
    calib_data = np.random.randn(100, 10).astype(np.float32)
    config = QConfig(
        is_static=True,
        weights_only=False,
        clip_ratio=0.95,
        reduce_range=False,
        mse=True,
        calibration_data=calib_data,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QInt8,
        weights_symmetric=True,
    )
    assert config.is_static is True
    assert config.weights_only is False
    assert config.clip_ratio == 0.95
    assert config.mse is True
    assert config.calibration_data is not None


def test_qconfig_dynamic_quantization():
    """Test dynamic quantization configuration."""
    config = QConfig(
        is_static=False,
        weights_only=False,
    )
    assert config.is_static is False
    assert config.weights_only is False


def test_qconfig_weights_only_quantization():
    """Test weights-only quantization configuration."""
    config = QConfig(
        weights_only=True,
        weights_dtype=QuantType.QInt4,
        weights_symmetric=True,
    )
    assert config.weights_only is True
    assert config.weights_dtype == QuantType.QInt4


def test_qconfig_with_all_parameters():
    """Test QConfig with all parameters specified."""
    gptq_config = GPTQConfig(block_size=512, percdamp=0.03, actorder=True)
    calib_data = np.random.randn(50, 20).astype(np.float32)

    config = QConfig(
        is_static=True,
        weights_only=False,
        clip_ratio=0.99,
        reduce_range=True,
        strategy=QuantizationStrategy.CHANNEL,
        mse=True,
        calibration_data=calib_data,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QInt8,
        weights_symmetric=False,
        algorithm=gptq_config,
    )

    assert config.is_static is True
    assert config.weights_only is False
    assert config.clip_ratio == 0.99
    assert config.reduce_range is True
    assert config.strategy == QuantizationStrategy.CHANNEL
    assert config.mse is True
    assert config.calibration_data is not None
    assert config.activations_dtype == QuantType.QUInt8
    assert config.activations_symmetric is True
    assert config.weights_dtype == QuantType.QInt8
    assert config.weights_symmetric is False
    assert config.algorithm is not None
    assert config.algorithm.block_size == 512


def test_qconfig_minimal_clip_ratio():
    """Test QConfig with very small but valid clip_ratio."""
    config = QConfig(clip_ratio=0.0001)
    assert config.clip_ratio == 0.0001


def test_qconfig_group_size_one():
    """Test group_size of 1."""
    config = QConfig(group_size=1, weights_only=True)
    assert config.group_size == 1
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_large_group_size():
    """Test very large group_size."""
    config = QConfig(group_size=10000, weights_only=True)
    assert config.group_size == 10000
    assert config.strategy == QuantizationStrategy.GROUP


def test_qconfig_all_symmetric():
    """Test configuration with all symmetric quantization."""
    config = QConfig(
        activations_symmetric=True,
        weights_symmetric=True,
    )
    assert config.activations_symmetric is True
    assert config.weights_symmetric is True


def test_qconfig_all_asymmetric():
    """Test configuration with all asymmetric quantization."""
    config = QConfig(
        activations_symmetric=False,
        weights_symmetric=False,
    )
    assert config.activations_symmetric is False
    assert config.weights_symmetric is False


def test_qconfig_empty_calibration_array():
    """Test QConfig with empty calibration data."""
    calib_data = np.array([])
    config = QConfig(calibration_data=calib_data)
    assert config.calibration_data is not None
    assert len(config.calibration_data) == 0


def test_qconfig_multidimensional_calibration_data():
    """Test QConfig with multidimensional calibration data."""
    calib_data = np.random.randn(10, 5, 3, 224, 224).astype(np.float32)
    config = QConfig(calibration_data=calib_data)
    assert config.calibration_data.shape == (10, 5, 3, 224, 224)


def test_qconfig_error_message_clip_ratio():
    """Test error message for invalid clip_ratio."""
    with pytest.raises(ValueError) as exc_info:
        QConfig(clip_ratio=1.5)
    assert "clip_ratio must be in (0.0, 1.0]" in str(exc_info.value)
    assert "1.5" in str(exc_info.value)


def test_qconfig_error_message_group_size_negative():
    """Test error message for invalid negative group_size."""
    with pytest.raises(ValueError) as exc_info:
        QConfig(group_size=-5)
    assert "Invalid group size" in str(exc_info.value)
    assert "-5" in str(exc_info.value)


def test_qconfig_error_message_gptq_group():
    """Test error message for GPTQ with GROUP strategy."""
    with pytest.raises(ValueError) as exc_info:
        QConfig(
            algorithm=GPTQConfig(),
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            weights_only=True,
        )
    assert "GPTQ algorithm only supports" in str(exc_info.value)


def test_qconfig_error_message_group_without_weights_only():
    """Test error message for GROUP strategy without weights_only."""
    with pytest.raises(ValueError) as exc_info:
        QConfig(group_size=128, weights_only=False)
    assert "Group quantization is only supported for weights_only" in str(exc_info.value)


def test_qconfig_is_pydantic_model():
    """Test that QConfig is a Pydantic BaseModel."""
    from pydantic import BaseModel

    config = QConfig()
    assert isinstance(config, BaseModel)


def test_qconfig_model_dump():
    """Test that QConfig can be dumped to dict."""
    config = QConfig(clip_ratio=0.95, weights_only=True)
    config_dict = config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["clip_ratio"] == 0.95
    assert config_dict["weights_only"] is True


def test_qconfig_model_dump_json():
    """Test that QConfig can be serialized to JSON."""
    config = QConfig(clip_ratio=0.95, weights_only=True, strategy="tensor")
    json_str = config.model_dump_json()
    assert isinstance(json_str, str)
    assert "0.95" in json_str
    assert "true" in json_str.lower()


def test_qconfig_calibration_params():
    """Test that calibration_params is a CalibrationParams instance."""
    config = QConfig(calibration_params={"momentum": 0.9, "num_samples": 50, "batch_size": 5})
    assert hasattr(config, "calibration_params")
    assert config.calibration_params.momentum == 0.9
    assert config.calibration_params.num_samples == 50
    assert config.calibration_params.batch_size == 5


def test_qconfig_calibration_params_default():
    """Test that default calibration_params is used when none provided."""
    config = QConfig()
    assert hasattr(config, "calibration_params")
    assert config.calibration_params.momentum == 0.0
    assert config.calibration_params.num_samples == 100
    assert config.calibration_params.method.value == "minmax"

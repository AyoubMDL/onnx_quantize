import numpy as np
import pytest

from onnx_quantize import (
    CalibrationMethod,
    GPTQConfig,
    HqqConfig,
    QActivationArgs,
    QConfig,
    QFormat,
    QuantizationStrategy,
    QuantType,
    QWeightArgs,
    SmoothQuantConfig,
)


class TestQConfig:
    def test_qconfig_defaults(self):
        config = QConfig(weights=QWeightArgs())
        assert config.format == "qdq"
        assert config.weights.dtype == QuantType.QInt8
        assert config.weights.strategy == QuantizationStrategy.TENSOR
        assert config.weights.symmetric is False
        assert config.input_activations is None
        assert config.output_activations is None

    def test_qconfig_valid_weights_only(self):
        # 4-bit weights only is allowed
        config = QConfig(weights=QWeightArgs(dtype=QuantType.QInt4))
        assert config.weights.dtype == QuantType.QInt4

        # Group quantization weights only is allowed
        config = QConfig(weights=QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=32))
        assert config.weights.strategy == QuantizationStrategy.GROUP
        assert config.weights.group_size == 32

    def test_qconfig_valid_static_quant(self):
        config = QConfig(
            weights=QWeightArgs(),
            input_activations=QActivationArgs(is_static=True),
            output_activations=QActivationArgs(is_static=True),
        )
        assert config.input_activations.is_static
        assert config.output_activations.is_static

    def test_qconfig_valid_dynamic_quant(self):
        # Default is dynamic (is_static=False)
        config = QConfig(
            weights=QWeightArgs(),
            input_activations=QActivationArgs(dtype=QuantType.QUInt8),
            output_activations=QActivationArgs(dtype=QuantType.QUInt8),
        )
        assert config.input_activations.is_static
        assert config.output_activations.is_static

    def test_qconfig_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid quantization format"):
            QConfig(weights=QWeightArgs(), format="invalid_format")

    def test_qconfig_calibration_params_default(self):
        config = QConfig(weights=QWeightArgs())
        assert config.calibration_params.momentum == 0.0
        assert config.calibration_params.num_samples == 100
        assert config.calibration_params.method.value == "minmax"

    def test_qconfig_calibration_params(self):
        config = QConfig(
            weights=QWeightArgs(),
            calibration_params={"momentum": 0.9, "num_samples": 50, "batch_size": 5},
        )
        assert config.calibration_params.method == CalibrationMethod.MINMAX
        assert config.calibration_params.momentum == 0.9
        assert config.calibration_params.num_samples == 50
        assert config.calibration_params.batch_size == 5

    def test_qconfig_calibration_data(self):
        calib_data = np.random.randn(10, 3, 224, 224).astype(np.float32)
        config = QConfig(weights=QWeightArgs(), calibration_data=calib_data)
        assert config.calibration_data.shape == (10, 3, 224, 224)

    @pytest.mark.parametrize("quant_type", [QuantType.QInt4, QuantType.QUInt4])
    def test_qconfig_4bit_with_input_activations_not_supported(self, quant_type):
        with pytest.raises(NotImplementedError, match="4-bit quantization is only supported"):
            QConfig(
                weights=QWeightArgs(dtype=quant_type),
                input_activations=QActivationArgs(),
            )

    @pytest.mark.parametrize("quant_type", [QuantType.QInt4, QuantType.QUInt4])
    def test_qconfig_4bit_with_output_activations_not_supported(self, quant_type):
        with pytest.raises(NotImplementedError, match="4-bit quantization is only supported"):
            QConfig(
                weights=QWeightArgs(dtype=quant_type),
                output_activations=QActivationArgs(),
            )

    @pytest.mark.parametrize("quant_type", [QuantType.QInt4, QuantType.QUInt4])
    def test_qconfig_4bit_with_both_activations_not_supported(self, quant_type):
        with pytest.raises(NotImplementedError, match="4-bit quantization is only supported"):
            QConfig(
                weights=QWeightArgs(dtype=quant_type),
                input_activations=QActivationArgs(),
                output_activations=QActivationArgs(),
            )

    def test_qconfig_group_quant_with_input_activations_not_supported(self):
        with pytest.raises(NotImplementedError, match="Group quantization is only supported"):
            QConfig(
                weights=QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=128),
                input_activations=QActivationArgs(),
            )

    def test_qconfig_group_quant_with_output_activations_not_supported(self):
        with pytest.raises(NotImplementedError, match="Group quantization is only supported"):
            QConfig(
                weights=QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=128),
                output_activations=QActivationArgs(),
            )

    def test_qconfig_group_quant_with_both_activations_not_supported(self):
        with pytest.raises(NotImplementedError, match="Group quantization is only supported"):
            QConfig(
                weights=QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=128),
                input_activations=QActivationArgs(),
                output_activations=QActivationArgs(),
            )

    @pytest.mark.parametrize("is_static", [True, False])
    def test_qconfig_mixed_static_dynamic_activations_not_supported(self, is_static):
        with pytest.raises(
            NotImplementedError, match="Both input and output activations must be either both"
        ):
            QConfig(
                weights=QWeightArgs(),
                input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=is_static),
                output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=not is_static),
            )

    def test_qconfig_qlinear_format(self):
        """Test that QLinear format can be created with valid configuration."""
        qconfig = QConfig(
            weights=QWeightArgs(),
            input_activations=QActivationArgs(is_static=True),
            output_activations=QActivationArgs(is_static=True),
            format="qlinear",
        )
        assert qconfig.format == QFormat.QLINEAR

    def test_qconfig_qlinear_missing_input_activations(self):
        with pytest.raises(ValueError, match="QLinear format requires both input and output"):
            QConfig(
                weights=QWeightArgs(),
                output_activations=QActivationArgs(is_static=True),
                format="qlinear",
            )

    def test_qconfig_qlinear_missing_output_activations(self):
        with pytest.raises(ValueError, match="QLinear format requires both input and output"):
            QConfig(
                weights=QWeightArgs(),
                input_activations=QActivationArgs(is_static=True),
                format="qlinear",
            )

    def test_qconfig_qlinear_dynamic_activations(self):
        with pytest.raises(
            ValueError, match="QLinear format requires both input and output activations.*static"
        ):
            QConfig(
                weights=QWeightArgs(),
                input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
                output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
                format="qlinear",
            )

    @pytest.mark.parametrize("input_dtype", [QuantType.QInt32, QuantType.QUInt32])
    def test_qconfig_qlinear_invalid_input_activation_dtype(self, input_dtype):
        with pytest.raises(
            ValueError, match="QLinear format supports only int8/uint8 for input activations"
        ):
            QConfig(
                weights=QWeightArgs(),
                input_activations=QActivationArgs(dtype=input_dtype, is_static=True),
                output_activations=QActivationArgs(is_static=True),
                format="qlinear",
            )

    @pytest.mark.parametrize("output_dtype", [QuantType.QInt32, QuantType.QUInt32])
    def test_qconfig_qlinear_invalid_output_activation_dtype(self, output_dtype):
        with pytest.raises(
            ValueError, match="QLinear format supports only int8/uint8 for output activations"
        ):
            QConfig(
                weights=QWeightArgs(),
                input_activations=QActivationArgs(is_static=True),
                output_activations=QActivationArgs(dtype=output_dtype, is_static=True),
                format="qlinear",
            )

    def test_qconfig_no_quantization_needed(self):
        config = QConfig()
        assert config.weights is None
        assert config.input_activations is None
        assert config.output_activations is None

    def test_qconfig_unsupported_activation_only_quantization(self):
        with pytest.raises(ValueError, match="Activation only quantization is not supported"):
            QConfig(
                input_activations=QActivationArgs(),
            )

        with pytest.raises(ValueError, match="Activation only quantization is not supported"):
            QConfig(
                output_activations=QActivationArgs(),
            )

        with pytest.raises(ValueError, match="Activation only quantization is not supported"):
            QConfig(
                input_activations=QActivationArgs(),
                output_activations=QActivationArgs(),
            )

    def test_qconfig_empty_preprocessors_default(self):
        config = QConfig(weights=QWeightArgs())
        assert config.preprocessors == ()

    def test_qconfig_single_preprocessor(self):
        config = QConfig(weights=QWeightArgs(), preprocessors=[SmoothQuantConfig(alpha=0.8)])
        assert len(config.preprocessors) == 1
        assert isinstance(config.preprocessors[0], SmoothQuantConfig)
        assert config.preprocessors[0].alpha == 0.8

    def test_qconfig_target_op_types_default(self):
        config = QConfig(weights=QWeightArgs())
        assert config.target_op_types == ("MatMul", "Gemm")

    def test_qconfig_target_op_types_removes_duplicates(self):
        """Test that duplicate target_op_types are removed."""
        config = QConfig(weights=QWeightArgs(), target_op_types=["MatMul", "Gemm", "MatMul"])
        assert config.target_op_types == ("Gemm", "MatMul")

    def test_qconfig_target_op_types(self):
        """Test that duplicate target_op_types are removed."""
        config = QConfig(weights=QWeightArgs(), target_op_types=["MatMul"])
        assert config.target_op_types == ("MatMul",)

    def test_qconfig_target_op_types_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported operator type.*Conv"):
            QConfig(weights=QWeightArgs(), target_op_types=["MatMul", "Conv"])


class TestQWeightArgs:
    def test_qweight_args_invalid_clip_ratio(self):
        with pytest.raises(ValueError, match="clip_ratio must be in"):
            QWeightArgs(clip_ratio=0.0)

        with pytest.raises(ValueError, match="clip_ratio must be in"):
            QWeightArgs(clip_ratio=-0.5)

        with pytest.raises(ValueError, match="clip_ratio must be in"):
            QWeightArgs(clip_ratio=1.5)

    def test_qweight_args_invalid_group_size(self):
        with pytest.raises(ValueError, match="Invalid group size"):
            QWeightArgs(group_size=-2)

        with pytest.raises(ValueError, match="Invalid group size"):
            QWeightArgs(group_size=-10)

    def test_qweight_args_group_size_without_group_strategy(self):
        with pytest.raises(ValueError, match="group_size requires strategy to be set to 'group'"):
            QWeightArgs(group_size=128, strategy=QuantizationStrategy.TENSOR)

        with pytest.raises(ValueError, match="group_size requires strategy to be set to 'group'"):
            QWeightArgs(group_size=128, strategy=QuantizationStrategy.CHANNEL)

    def test_qweight_args_group_strategy_without_group_size(self):
        with pytest.raises(ValueError, match="strategy .* requires group_size"):
            QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=None)

        with pytest.raises(ValueError, match="strategy .* requires group_size"):
            QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=0)

        with pytest.raises(ValueError, match="strategy .* requires group_size"):
            QWeightArgs(strategy=QuantizationStrategy.GROUP, group_size=-1)

    def test_qweight_args_gptq_with_group_strategy(self):
        args = QWeightArgs(
            algorithm=GPTQConfig(),
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        )
        assert isinstance(args.algorithm, GPTQConfig)
        assert args.strategy == QuantizationStrategy.GROUP
        assert args.group_size == 128

    def test_qweight_args_hqq_valid_config(self):
        args = QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy=QuantizationStrategy.GROUP,
            group_size=32,
            symmetric=False,
            algorithm=HqqConfig(),
        )
        assert isinstance(args.algorithm, HqqConfig)
        assert args.dtype == QuantType.QUInt4
        assert args.strategy == QuantizationStrategy.GROUP
        assert args.symmetric is False
        assert args.group_size == 32
        assert args.zp_dtype == args.scale_dtype

    @pytest.mark.parametrize("quant_type", [QuantType.QInt8, QuantType.QUInt8])
    def test_qweight_args_hqq_invalid_dtype(self, quant_type):
        with pytest.raises(ValueError, match="HQQ only supports uint4 weight type"):
            QWeightArgs(
                dtype=quant_type,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
                algorithm=HqqConfig(),
            )

    def test_qweight_args_hqq_invalid_symmetric(self):
        with pytest.raises(ValueError, match="HQQ only supports asymmetric quantization"):
            QWeightArgs(
                dtype=QuantType.QUInt4,
                strategy=QuantizationStrategy.GROUP,
                group_size=32,
                symmetric=True,
                algorithm=HqqConfig(),
            )

    @pytest.mark.parametrize("strategy", ["tensor", "channel"])
    def test_qweight_args_hqq_invalid_strategy(self, strategy):
        with pytest.raises(ValueError, match="HQQ only supports 'group' quantization strategy"):
            QWeightArgs(
                dtype=QuantType.QUInt4,
                strategy=strategy,
                algorithm=HqqConfig(),
            )

    def test_qweight_args_hqq_invalid_group_size(self):
        # Group size less than 16
        with pytest.raises(ValueError, match="HQQ requires group_size to be greater than 16"):
            QWeightArgs(
                dtype=QuantType.QUInt4,
                strategy=QuantizationStrategy.GROUP,
                group_size=8,
                algorithm=HqqConfig(),
            )

        # Group size not power of 2
        with pytest.raises(ValueError, match="HQQ requires group_size to be greater than 16"):
            QWeightArgs(
                dtype=QuantType.QUInt4,
                strategy=QuantizationStrategy.GROUP,
                group_size=48,
                algorithm=HqqConfig(),
            )

    def test_qweight_args_hqq_valid_group_sizes(self):
        for group_size in [16, 32, 64, 128, 256]:
            args = QWeightArgs(
                dtype=QuantType.QUInt4,
                strategy=QuantizationStrategy.GROUP,
                group_size=group_size,
                symmetric=False,
                algorithm=HqqConfig(),
            )
            assert args.group_size == group_size

    def test_qweight_args_hqq_scale_zp_same_dtype(self):
        # Even if we provide different dtypes, it should be corrected
        args = QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy=QuantizationStrategy.GROUP,
            group_size=32,
            symmetric=False,
            algorithm=HqqConfig(),
            scale_dtype=np.float32,
        )
        # zp_dtype should be set to match scale_dtype
        assert args.scale_dtype == args.zp_dtype

    def test_qweight_args_hqq_custom_parameters(self):
        args = QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy=QuantizationStrategy.GROUP,
            group_size=64,
            symmetric=False,
            algorithm=HqqConfig(lp_norm=0.5, beta=5.0, kappa=1.05, iters=10, early_stop=False),
        )
        assert isinstance(args.algorithm, HqqConfig)
        assert args.algorithm.lp_norm == 0.5
        assert args.algorithm.beta == 5.0
        assert args.algorithm.kappa == 1.05
        assert args.algorithm.iters == 10
        assert args.algorithm.early_stop is False

    def test_qweight_args_invalid_scale_dtype(self):
        with pytest.raises(ValueError, match="Only float32 scale dtype is currently supported."):
            QWeightArgs(scale_dtype=np.float16)

        with pytest.raises(ValueError, match="Only float32 scale dtype is currently supported."):
            QWeightArgs(scale_dtype=np.int32)

    def test_qweight_default_values(self):
        args = QWeightArgs()
        assert args.dtype == QuantType.QInt8
        assert args.strategy == QuantizationStrategy.TENSOR
        assert args.symmetric is False
        assert args.scale_dtype == np.dtype(np.float32)
        assert args.clip_ratio == 1.0
        assert args.mse is False
        assert args.group_size is None

    def test_qweight_strategy_from_string(self):
        config = QConfig(weights=QWeightArgs(strategy="tensor"))
        assert config.weights.strategy == QuantizationStrategy.TENSOR

        config = QConfig(weights=QWeightArgs(strategy="channel"))
        assert config.weights.strategy == QuantizationStrategy.CHANNEL

        config = QConfig(weights=QWeightArgs(strategy="group", group_size=32))
        assert config.weights.strategy == QuantizationStrategy.GROUP

    def test_qweight_strategy_inference(self):
        # Tensor quantization (default)
        args = QWeightArgs(group_size=None)
        assert args.strategy == QuantizationStrategy.TENSOR

        # Channel quantization
        args = QWeightArgs(group_size=-1)
        assert args.strategy == QuantizationStrategy.CHANNEL

        # Group quantization
        args = QWeightArgs(group_size=32)
        assert args.strategy == QuantizationStrategy.GROUP

    def test_qweight_string_inputs(self):
        # Note: strategy needs to match group_size or be inferable.
        # If we pass explicit strategy, validation will check if it matches group_size.
        # Here we test conversion from string.
        # group_size -1 and strategy "channel" are consistent.
        args = QWeightArgs(dtype="int4", strategy="channel", group_size=-1)
        assert args.dtype == QuantType.QInt4
        assert args.strategy == QuantizationStrategy.CHANNEL

    def test_qweight_scale_dtype_valid(self):
        args = QWeightArgs(scale_dtype=np.float32)
        assert args.scale_dtype == np.dtype(np.float32)


class TestQActivationArgs:
    def test_qactivation_args_strategy_not_supported(self):
        with pytest.raises(NotImplementedError, match="Activation quantization only supports"):
            QActivationArgs(strategy=QuantizationStrategy.CHANNEL)

        with pytest.raises(NotImplementedError, match="Activation quantization only supports"):
            QActivationArgs(strategy=QuantizationStrategy.GROUP, group_size=128)

    @pytest.mark.parametrize("quant_type", [QuantType.QInt4, QuantType.QUInt4])
    def test_qactivation_args_qint4_not_supported(self, quant_type):
        with pytest.raises(NotImplementedError, match="4-bit quantization is not supported"):
            QActivationArgs(dtype=quant_type)

    @pytest.mark.parametrize("quant_type", [QuantType.QInt32, QuantType.QInt8])
    def test_qactivation_args_dynamic_qint_not_supported(self, quant_type):
        with pytest.raises(
            NotImplementedError, match="Dynamic activation quantization only supports"
        ):
            QActivationArgs(dtype=quant_type, is_static=False)

    def test_qactivation_defaults(self):
        qargs = QActivationArgs()
        assert qargs.dtype == QuantType.QInt8
        assert qargs.symmetric is False
        assert qargs.strategy == QuantizationStrategy.TENSOR
        assert qargs.is_static is True
        assert qargs.group_size is None

    def test_qactivation_valid_static(self):
        qargs = QActivationArgs(is_static=True)
        assert qargs.is_static is True
        assert qargs.strategy == QuantizationStrategy.TENSOR

    def test_qactivation_valid_dynamic_uint8(self):
        qargs = QActivationArgs(dtype=QuantType.QUInt8, is_static=False)
        assert qargs.dtype == QuantType.QUInt8
        assert qargs.is_static is False

    def test_qactivation_string_inputs(self):
        qargs = QActivationArgs(dtype="uint8")
        assert qargs.dtype == QuantType.QUInt8
        assert qargs.symmetric is False

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from onnx_quantize.core._calibration.base import CalibrationParams
from onnx_quantize.core._dtypes import QuantType


class QuantizationStrategy(str, Enum):
    """Enum storing quantization strategy options."""

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


class QFormat(str, Enum):
    """Enum storing quantization format options."""

    QDQ = "qdq"
    QLINEAR = "qlinear"


class GPTQConfig(BaseModel):
    """GPTQConfig is the configuration class handling all the GPTQ quantization parameters.

    Args:
        block_size (int, optional): GPTQ block size. Defaults to 128.
        percdamp (float, optional): GPTQ percent of damping. Defaults to 0.01.
        group_size (int, optional): GPTQ group size. Defaults to -1.
        actorder (bool, optional): GPTQ activation order. Defaults to False.
    """

    block_size: int = 128
    percdamp: float = 0.01
    group_size: int = -1
    actorder: bool = False


AlgorithmConfig = GPTQConfig | None


class _BaseArgs(BaseModel):
    # Allow arbitrary types for np dtype
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: QuantType | str = QuantType.QInt8
    symmetric: bool = False
    group_size: int | None = Field(
        default=None, description=">0: group quant, -1: channel quant, None: tensor quant"
    )
    strategy: QuantizationStrategy | str | None = None
    scale_dtype: np.dtype = Field(default=np.dtype(np.float32))
    reduce_range: bool = False

    @field_validator("dtype", mode="before")
    def validate_dtype_before(cls, value) -> QuantType:
        if isinstance(value, str):
            return QuantType.from_string(value)

        return value

    @field_validator("group_size", mode="before")
    def validate_group(cls, value) -> int | None:
        if value is None:
            return value

        if value < -1:
            raise ValueError(
                f"Invalid group size {value}. Use group_size > 0 for "
                "strategy='group' and group_size = -1 for 'per_channel'"
            )

        return value

    @field_validator("strategy", mode="before")
    def validate_strategy_before(cls, value) -> QuantizationStrategy:
        if isinstance(value, str):
            return QuantizationStrategy(value.lower())

        return value

    @field_validator("scale_dtype", mode="before")
    def validate_scale_dtype_before(cls, value) -> np.dtype:
        # Convert numpy type to dtype if needed
        if isinstance(value, type) and issubclass(value, np.generic):
            return np.dtype(value)

        if not isinstance(value, np.dtype):
            return np.dtype(value)

        return value

    @field_validator("scale_dtype", mode="after")
    def validate_scale_dtype_after(cls, value) -> np.dtype:
        # TODO: Support float16
        if value != np.float32:
            raise ValueError("Only float32 scale dtype is currently supported.")

        return value

    @model_validator(mode="after")
    def validate_model_after(self: _BaseArgs) -> _BaseArgs:
        # extract user-passed values from dictionary
        strategy = self.strategy
        group_size = self.group_size

        # infer strategy
        if strategy is None:
            if group_size is None:
                strategy = QuantizationStrategy.TENSOR
            elif group_size > 0:
                strategy = QuantizationStrategy.GROUP
            elif group_size == -1:
                strategy = QuantizationStrategy.CHANNEL
            else:
                raise ValueError(
                    f"Invalid group size {group_size}. Use group_size > 0 for "
                    "strategy='group' and group_size = -1 for 'channel'"
                )

        # validate group strategy
        if strategy == QuantizationStrategy.GROUP:
            if group_size is None or group_size <= 0:
                raise ValueError(
                    f"strategy {strategy} requires group_size to be set to a positive value."
                )

        if group_size is not None and group_size > 0 and strategy != QuantizationStrategy.GROUP:
            raise ValueError("group_size requires strategy to be set to 'group'.")

        # write back modified values
        self.strategy = strategy
        return self


class QWeightArgs(_BaseArgs):
    """QWeightArgs is the configuration class handling all the weight quantization parameters.

    Args:
        clip_ratio (float, optional): Ratio for clipping weights before quantization.
            Defaults to 1.0.
        mse (bool, optional): Whether to use MSE-based quantization. Defaults to False.
        algorithm (AlgorithmConfig | None, optional): Algorithm-specific configuration.
            Defaults to None.
    """

    clip_ratio: float = 1.0
    mse: bool = False
    algorithm: AlgorithmConfig | None = None

    @field_validator("clip_ratio", mode="after")
    def validate_clip_ratio(cls, value) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"clip_ratio must be in (0.0, 1.0], got {value}")
        return value

    @model_validator(mode="after")
    def validate_model_after(self: QWeightArgs) -> QWeightArgs:
        self = super().validate_model_after()
        if isinstance(self.algorithm, GPTQConfig) and self.strategy not in {
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        }:
            raise NotImplementedError(
                "GPTQ algorithm only supports 'tensor' or 'channel' quantization."
            )
        return self


class QActivationArgs(_BaseArgs):
    """The configuration class handling the activation quantization parameters.

    Args:
        is_static (bool, optional): Whether the activation quantization is static (calibrated)
            or dynamic. Defaults to True.
    """

    is_static: bool = True

    # Validate that strategy should always be tensor for activations
    @field_validator("strategy", mode="after")
    def validate_strategy(cls, value) -> QuantizationStrategy:
        if value is not None and value != QuantizationStrategy.TENSOR:
            raise NotImplementedError("Activation quantization only supports 'tensor' strategy.")

        return QuantizationStrategy.TENSOR

    @field_validator("dtype", mode="after")
    def validate_dtype(cls, value) -> QuantType:
        if value in {QuantType.QInt4, QuantType.QUInt4}:
            raise NotImplementedError("4-bit quantization is not supported for activations.")

        return value

    @model_validator(mode="after")
    def validate_model_after(self: QActivationArgs) -> QActivationArgs:
        # TODO: add support for other dtypes with dynamic quantization
        if not self.is_static and self.dtype != QuantType.QUInt8:
            raise NotImplementedError("Dynamic activation quantization only supports uint8 dtype.")

        return super().validate_model_after()


class QConfig(BaseModel):
    """QConfig is the main configuration class handling all the quantization parameters.

    Args:
        weights (QWeightArgs | None, optional): Weight quantization parameters.
            Defaults to None.
        input_activations (QActivationArgs | None, optional): Input activation quantization
            parameters. Defaults to None.
        output_activations (QActivationArgs | None, optional): Output activation quantization
            parameters. Defaults to None.
        format (QFormat | str, optional): Quantization format. Defaults to QFormat.QDQ.
        calibration_params (CalibrationParams | None, optional): Calibration parameters.
            Defaults to CalibrationParams().
        calibration_data (np.ndarray | None, optional): Calibration data for static quantization.
            Defaults to None.
    """

    weights: QWeightArgs | None = None
    input_activations: QActivationArgs | None = None
    output_activations: QActivationArgs | None = None
    format: QFormat | str = QFormat.QDQ

    # Same calibration data for both weights and activations
    # Needed for activation and also for weights only quantization with GPTQ
    calibration_params: CalibrationParams | None = Field(default_factory=CalibrationParams)
    calibration_data: np.ndarray | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("format", mode="before")
    def validate_format(cls, value) -> QFormat:
        if isinstance(value, str):
            try:
                return QFormat(value.lower())
            except ValueError:
                valid_formats = [f.value for f in QFormat]
                raise ValueError(  # noqa: B904
                    f"Invalid quantization format '{value}'. Valid formats are: {valid_formats}"
                )

        return value

    @field_validator("calibration_params", mode="before")
    def validate_calibration_params(cls, value) -> CalibrationParams:
        if isinstance(value, dict):
            return CalibrationParams(**value)

        return value

    @model_validator(mode="after")
    def validate_model_after(self: QConfig) -> QConfig:
        # Check if everything is None
        if (
            self.weights is None
            and self.input_activations is None
            and self.output_activations is None
        ):
            return self

        if self.weights is None:
            raise ValueError("Activation only quantization is not supported.")

        weights_only = self.input_activations is None and self.output_activations is None
        if (not weights_only) and self.weights.dtype in {QuantType.QInt4, QuantType.QUInt4}:
            raise NotImplementedError(
                "4-bit quantization is only supported for weights_only quantization."
            )

        if self.weights.strategy == QuantizationStrategy.GROUP and not weights_only:
            raise NotImplementedError(
                "Group quantization is only supported for weights_only quantization."
            )

        # Ensure that both input and output activations are either static or dynamic
        if self.input_activations is not None and self.output_activations is not None:
            if self.input_activations.is_static != self.output_activations.is_static:
                raise NotImplementedError(
                    "Both input and output activations must be either both static or dynamic."
                )

        if self.format != QFormat.QDQ:
            raise NotImplementedError("Only QDQ format is currently supported.")

        return self

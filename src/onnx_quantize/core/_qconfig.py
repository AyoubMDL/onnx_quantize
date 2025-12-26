from __future__ import annotations


__all__ = ["QConfig", "QuantizationStrategy", "GPTQConfig"]

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from onnx_quantize.core._dtypes import QuantType


class QuantizationStrategy(str, Enum):
    """Enum storing quantization strategy options."""

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


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


class QConfig(BaseModel):
    """QConfig is the configuration class handling all the quantization parameters.

    Args:
        is_static (`bool`, optional): Whether it is static or dynamic quantization.
            Defaults to `True`.
        weights_only (`bool`, optional): Whether to quantize only weights or not.
            Defaults to `False`.
        clip_ratio (`float`, optional): Percentile of clip. Must be in (0.0, 1.0].
            Defaults to `1.0`.
        reduce_range (`bool`, optional): Whether to use reduced range for quantization.
            Defaults to `False`.
        group_size (`int | None`, optional): Quantization granularity. Defaults to `None`.
            - `None`: tensor-wise quantization
            - `-1`: per-channel quantization
            - `> 0`: group quantization with the specified group size
        strategy (`QuantizationStrategy | str | None`, optional): Quantization strategy.
            Defaults to `None`. If not specified, will be inferred from `group_size`.
            - `tensor`: tensor-wise quantization
            - `channel`: per-channel quantization
            - `group`: group quantization
        mse (`bool`, optional): Whether to use MSE minimization to compute
            quantization parameters. Defaults to `False`.
        calibration_data (`np.ndarray | None`, optional): Calibration data for
            static quantization. Defaults to `None`.
        activations_dtype (`QuantType | str`, optional):
            The quantization data type to use for the activations.
            Defaults to `QuantType.QUInt8`.
        activations_symmetric (`bool`, optional):
            Whether to apply symmetric quantization on the activations.
            Defaults to `False`.
        weights_dtype (`QuantType | str`, optional):
            The quantization data type to use for the weights.
            Defaults to `QuantType.QInt8`.
        weights_symmetric (`bool`, optional):
            Whether to apply symmetric quantization on the weights.
            Defaults to `True`.
        algorithm (`AlgorithmConfig | None`, optional):
            Advanced quantization algorithm configuration (e.g., `GPTQConfig`).
            Defaults to `None`. GPTQ algorithm only supports 'tensor' or 'channel'
            quantization strategies.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_static: bool = True
    weights_only: bool = False
    clip_ratio: float = 1.0
    reduce_range: bool = False
    group_size: int | None = Field(
        default=None, description=">0: group quant, -1: channel quant, None: tensor quant"
    )
    strategy: QuantizationStrategy | str | None = None
    mse: bool = False
    calibration_data: np.ndarray | None = None
    activations_dtype: QuantType = QuantType.QUInt8
    activations_symmetric: bool = False
    weights_dtype: QuantType | str = QuantType.QInt8
    weights_symmetric: bool = True
    algorithm: AlgorithmConfig | None = None

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

    @field_validator("weights_dtype", mode="before")
    def validate_weights_dtype(cls, value) -> QuantType | None:
        if isinstance(value, str):
            return QuantType.from_string(value)

        return value

    @field_validator("activations_dtype", mode="before")
    def validate_activations_dtype(cls, value) -> QuantType | None:
        if isinstance(value, str):
            return QuantType.from_string(value)

        return value

    @field_validator("strategy", mode="before")
    def validate_strategy(cls, value) -> QuantizationStrategy | None:
        if isinstance(value, str):
            return QuantizationStrategy(value.lower())

        return value

    @field_validator("clip_ratio", mode="after")
    def validate_clip_ratio(cls, value) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"clip_ratio must be in (0.0, 1.0], got {value}")
        return value

    @model_validator(mode="after")
    def validate_model_after(self: QConfig) -> QConfig:
        # extract user-passed values from dictionary
        strategy = self.strategy
        group_size = self.group_size
        algorithm = self.algorithm

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

        if self.activations_dtype in {QuantType.QInt4, QuantType.QUInt4}:
            raise ValueError("4-bit quantization is not supported for activations.")

        if self.weights_dtype in {QuantType.QInt4, QuantType.QUInt4} and not self.weights_only:
            raise ValueError("4-bit quantization is only supported for weights_only quantization.")

        if isinstance(algorithm, GPTQConfig) and strategy not in {
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        }:
            raise ValueError("GPTQ algorithm only supports 'tensor' or 'channel' quantization.")

        if strategy == QuantizationStrategy.GROUP and not self.weights_only:
            raise ValueError("Group quantization is only supported for weights_only quantization.")

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

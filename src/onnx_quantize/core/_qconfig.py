from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from onnx_quantize.core._calibration.base import CalibrationParams
from onnx_quantize.core._dtypes import QuantType


if TYPE_CHECKING:
    import onnx_ir as ir


logger = logging.getLogger(__name__)

_SUPPORTED_OP_TYPES = ("MatMul", "Gemm")


class QuantizationStrategy(str, Enum):
    """Enum storing quantization strategy options."""

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


class QFormat(str, Enum):
    """Enum storing quantization format options."""

    QDQ = "qdq"
    QLINEAR = "qlinear"


class AlgorithmConfig(BaseModel):
    """Base class for weight-quantization algorithm configurations.

    Subclass this to add a new weight-quantization algorithm. A subclass lives
    entirely in its own module (next to its quantization kernel) and plugs into the
    pipeline without any change to this core configuration code:

    * declare an ``algorithm_type`` ``Literal`` field whose default is the tag used
      to reconstruct the config from a serialized ``QConfig``,
    * decorate the class with :func:`register_algorithm_config`,
    * set the ``requires_calibration`` class variable to ``True`` when the algorithm
      needs input activations,
    * implement :meth:`quantize_weights`, and
    * optionally override :meth:`validate_weight_args` to enforce extra constraints.
    """

    # Whether the algorithm needs input activations to be collected during calibration.
    requires_calibration: ClassVar[bool] = False

    def validate_weight_args(self, weight_args: QWeightArgs) -> None:
        """Validate/adjust the enclosing ``QWeightArgs``. No-op by default."""

    def quantize_weights(
        self, w: ir.Value, qconfig: QConfig, out: ir.Value | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize weight tensor ``w`` and return ``(q_weight, scale, zero_point)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement quantize_weights().")


class PreProcessingConfig(BaseModel):
    """Base class for pre-processing configurations.

    Subclass this to add a new pre-processing pass. A subclass lives entirely in its
    own module (next to its pass implementation) and plugs into the pipeline without
    any change to this core configuration code:

    * declare a ``preprocessing_type`` ``Literal`` field whose default is the tag used
      to reconstruct the config from a serialized ``QConfig``,
    * decorate the class with :func:`register_preprocessing_config`,
    * set the ``requires_calibration`` / ``requires_post_calibration`` class variables
      as appropriate, and
    * implement :meth:`build_pass`.
    """

    # Whether the pre-processing needs input activations collected during calibration.
    requires_calibration: ClassVar[bool] = True
    # Whether the model must be re-calibrated after this pre-processing runs.
    requires_post_calibration: ClassVar[bool] = True

    def build_pass(self, qconfig: QConfig) -> ir.passes.InPlacePass:
        """Build the IR pass that applies this pre-processing."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_pass().")


_ALGORITHM_REGISTRY: dict[str, type[AlgorithmConfig]] = {}
_PREPROCESSING_REGISTRY: dict[str, type[PreProcessingConfig]] = {}


def register_algorithm_config(cls: type[AlgorithmConfig]) -> type[AlgorithmConfig]:
    """Register an :class:`AlgorithmConfig` subclass under its ``algorithm_type`` tag."""
    field = cls.model_fields.get("algorithm_type")
    if field is None:
        raise TypeError(f"{cls.__name__} must declare an 'algorithm_type' field to be registered.")
    _ALGORITHM_REGISTRY[field.default] = cls
    return cls


def register_preprocessing_config(cls: type[PreProcessingConfig]) -> type[PreProcessingConfig]:
    """Register a :class:`PreProcessingConfig` subclass under its ``preprocessing_type`` tag."""
    field = cls.model_fields.get("preprocessing_type")
    if field is None:
        raise TypeError(
            f"{cls.__name__} must declare a 'preprocessing_type' field to be registered."
        )
    _PREPROCESSING_REGISTRY[field.default] = cls
    return cls


def _default_algorithm_config() -> AlgorithmConfig:
    """Return the default weight-quantization algorithm (round-to-nearest)."""
    from onnx_quantize.core._algorithms.rtn import RTNConfig

    return RTNConfig()


def _resolve_algorithm_config(value):
    """Coerce a raw value into a concrete :class:`AlgorithmConfig` instance.

    ``None`` becomes the default algorithm, instances pass through untouched, and a
    mapping is dispatched to the registered subclass via its ``algorithm_type`` tag
    (used when reconstructing a ``QConfig`` from serialized node metadata).
    """
    if value is None:
        return _default_algorithm_config()
    if isinstance(value, AlgorithmConfig):
        return value
    if isinstance(value, dict):
        tag = value.get("algorithm_type")
        if tag not in _ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm_type {tag!r}. Registered: {sorted(_ALGORITHM_REGISTRY)}"
            )
        return _ALGORITHM_REGISTRY[tag](**value)
    return value


def _resolve_preprocessing_config(value):
    """Coerce a raw value into a concrete :class:`PreProcessingConfig` instance.

    Instances pass through untouched; a mapping is dispatched to the registered
    subclass via its ``preprocessing_type`` tag.
    """
    if isinstance(value, PreProcessingConfig):
        return value
    if isinstance(value, dict):
        tag = value.get("preprocessing_type")
        if tag not in _PREPROCESSING_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing_type {tag!r}. Registered: {sorted(_PREPROCESSING_REGISTRY)}"
            )
        return _PREPROCESSING_REGISTRY[tag](**value)
    return value


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

    # Zero point field is set during validation
    zp_dtype: np.dtype = Field(default=None, init=False)
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

        # Define zero point dtype
        if self.zp_dtype is None:
            self.zp_dtype = self.dtype.np_dtype

        # write back modified values
        self.strategy = strategy
        return self


class QWeightArgs(_BaseArgs):
    """QWeightArgs is the configuration class handling all the weight quantization parameters.

    Args:
        clip_ratio (float, optional): Ratio for clipping weights before quantization.
            Defaults to 1.0.
        mse (bool, optional): Whether to use MSE-based quantization. Defaults to False.
        algorithm (AlgorithmConfig, optional): Weight-quantization algorithm configuration.
            ``None`` is coerced to the default round-to-nearest ``RTNConfig``.
    """

    clip_ratio: float = 1.0
    mse: bool = False
    algorithm: SerializeAsAny[AlgorithmConfig] = Field(default_factory=_default_algorithm_config)

    @field_validator("algorithm", mode="before")
    def validate_algorithm(cls, value):
        return _resolve_algorithm_config(value)

    @field_validator("clip_ratio", mode="after")
    def validate_clip_ratio(cls, value) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"clip_ratio must be in (0.0, 1.0], got {value}")
        return value

    @model_validator(mode="after")
    def validate_model_after(self: QWeightArgs) -> QWeightArgs:
        # Algorithms plug their own constraints/adjustments in via this hook.
        self.algorithm.validate_weight_args(self)

        return super().validate_model_after()


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
    r"""QConfig is the main configuration class handling all the quantization parameters.

    Args:
        target_op_types (Sequence[str], optional): Sequence of target operator types to quantize.
            Defaults to {"MatMul", "Gemm"}.
        weights (QWeightArgs | None, optional): Weight quantization parameters.
            Defaults to None.
        input_activations (QActivationArgs | None, optional): Input activation quantization
            parameters. Defaults to None.
        output_activations (QActivationArgs | None, optional): Output activation quantization
            parameters. Defaults to None.
        format (QFormat | str, optional): Quantization format. Defaults to QFormat.QDQ.
        calibration_params (CalibrationParams | None, optional): Calibration parameters.
            Defaults to CalibrationParams().
        calibration_data (np.ndarray | dict[str, np.ndarray] | None, optional): Calibration data
            for static quantization. A single array is mapped to the model's first input; a dict
            maps input names to arrays (required for multi-input models). Defaults to None.
        preprocessors (Sequence[PreProcessingConfig], optional): Sequence of pre-processing
            configurations to apply before quantization. Defaults to an empty tuple.
        ignore (Sequence[str], optional): Regex patterns matched against node names with
            ``re.search`` (e.g. "lm_head", "embed", r"^layers\.0\."). Matching nodes are
            skipped from quantization. Defaults to an empty tuple.
    """

    target_op_types: Sequence[str] = Field(default_factory=lambda: _SUPPORTED_OP_TYPES)
    weights: QWeightArgs | None = None
    input_activations: QActivationArgs | None = None
    output_activations: QActivationArgs | None = None
    format: QFormat | str = QFormat.QDQ

    # Same calibration data for both weights and activations
    # Needed for activation and also for weights only quantization with GPTQ
    calibration_params: CalibrationParams | None = Field(default_factory=CalibrationParams)
    calibration_data: np.ndarray | dict[str, np.ndarray] | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Preprocessors
    preprocessors: Sequence[SerializeAsAny[PreProcessingConfig]] = Field(default_factory=tuple)

    # Patterns of node names to skip from quantization
    ignore: Sequence[str] = Field(default_factory=tuple)

    @field_validator("target_op_types", mode="before")
    def validate_target_op_types(cls, value) -> Sequence[str]:
        # Remove duplicates and convert to tuple for immutability
        return tuple(sorted(set(value)))

    @field_validator("ignore", mode="before")
    def validate_ignore(cls, value) -> Sequence[str]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        return tuple(value)

    @field_validator("preprocessors", mode="before")
    def validate_preprocessors(cls, value) -> Sequence[PreProcessingConfig]:
        # Coerce each entry to its concrete config (dispatching mappings via the
        # registry) so subtypes survive the QConfig -> node-metadata round-trip.
        if value is None:
            return ()
        return tuple(_resolve_preprocessing_config(item) for item in value)

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

    def _check_qlinear_format_constraints(self) -> None:
        # TODO: some checks overlaps with other parts, refactor later
        if self.input_activations is None or self.output_activations is None:
            raise ValueError(
                "QLinear format requires both input and output activation quantization."
            )

        if not (self.input_activations.is_static and self.output_activations.is_static):
            raise ValueError(
                "QLinear format requires both input and output activations "
                "quantization to be static."
            )

        if self.weights.strategy == QuantizationStrategy.GROUP:
            raise NotImplementedError(
                "QLinear format does not support grouped weight quantization."
            )

        # dtypes for weights and activations should be int8 or uint8
        valid_dtypes = {QuantType.QInt8, QuantType.QUInt8}
        if self.weights.dtype not in valid_dtypes:
            raise ValueError(
                f"QLinear format supports only int8/uint8 for weights, got {self.weights.dtype}."
            )

        if self.input_activations.dtype not in valid_dtypes:
            raise ValueError(
                f"QLinear format supports only int8/uint8 for input activations, got "
                f"{self.input_activations.dtype}."
            )

        if self.output_activations.dtype not in valid_dtypes:
            raise ValueError(
                f"QLinear format supports only int8/uint8 for output activations, got "
                f"{self.output_activations.dtype}."
            )

    @model_validator(mode="after")
    def validate_model_after(self: QConfig) -> QConfig:
        for op_type in self.target_op_types:
            if op_type not in _SUPPORTED_OP_TYPES:
                raise ValueError(
                    f"Unsupported operator type '{op_type}' in target_op_types. "
                    f"Supported operator types are: {_SUPPORTED_OP_TYPES}"
                )

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

        # TODO: Maybe allow weights to be 4bits with 8bits activations
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

        if self.format == QFormat.QLINEAR:
            self._check_qlinear_format_constraints()

        return self

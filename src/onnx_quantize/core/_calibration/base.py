__all__ = ["CalibrationMethod", "CalibrationParams"]

import abc
import dataclasses
import enum

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class CalibrationMethod(enum.Enum):
    """Calibration method enum."""

    MINMAX = "minmax"


class CalibrationParams(BaseModel):
    """Calibration parameters for quantization.

    Args:
        method (CalibrationMethod, optional): Calibration method to use.
            Defaults to CalibrationMethod.MINMAX.
        num_samples (int, optional): Number of samples to use for calibration.
            Defaults to 100.
        batch_size (int, optional): Batch size for processing calibration samples.
            Defaults to 10.
        momentum (float, optional): Momentum for moving average calculations.
            Must be in the range [0, 1). Defaults to 0.0.
    """

    # Forbid extra fields
    model_config = ConfigDict(extra="forbid")

    method: CalibrationMethod | str = CalibrationMethod.MINMAX
    num_samples: int = 100
    batch_size: int = 10
    momentum: float = 0.0

    @field_validator("method", mode="before")
    def validate_method(cls, value) -> CalibrationMethod:
        if isinstance(value, str):
            try:
                return CalibrationMethod(value)
            except ValueError:
                valid_methods = [m.value for m in CalibrationMethod]
                raise ValueError(  # noqa: B904
                    f"Invalid calibration method '{value}'. Valid methods are: {valid_methods}"
                )
        return value

    @field_validator("momentum", mode="after")
    def validate_momentum(cls, value) -> float:
        if not 0 <= value < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {value}")
        return value

    @field_validator("num_samples", "batch_size", mode="after")
    def validate_positive(cls, value, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {value}")
        return value


@dataclasses.dataclass
class CalibrationData:
    """Container for calibration statistics collected during model inference.

    Attributes:
        min_val: Minimum value observed
        max_val: Maximum value observed
    """

    min_val: np.ndarray
    max_val: np.ndarray


class Calibrator(abc.ABC):
    """Abstract base class for calibration methods.

    Subclasses must implement:
    - collect(): Process activation data and update statistics
    - compute_range(): Compute final min/max values from collected statistics
    """

    def __init__(self):
        """Initialize calibrator with empty statistics storage."""
        self.data: dict[str, CalibrationData] = {}

    @abc.abstractmethod
    def collect(self, name: str, array: np.ndarray) -> None:
        """Collect calibration statistics from an array.

        Args:
            name (str): Unique identifier for the array (e.g., node input name)
            array (np.ndarray): Activation array to collect statistics from
        """
        pass

    @abc.abstractmethod
    def compute_range(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute the final quantization range from collected statistics.

        Args:
            name (str): Unique identifier for the array
        Returns:
            tuple: (min_val, max_val) for quantization
        """
        pass

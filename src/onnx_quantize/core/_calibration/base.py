__all__ = ["CalibrationMethod"]

import abc
import dataclasses
import enum

import numpy as np


class CalibrationMethod(enum.Enum):
    """Calibration method enum."""

    MINMAX = "minmax"


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

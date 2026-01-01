import numpy as np

from onnx_quantize.core._calibration.base import CalibrationData, Calibrator


class MinMaxCalibrator(Calibrator):
    """MinMax calibrator that tracks minimum and maximum values.

    This is the simplest calibration method that directly uses the observed
    minimum and maximum values from the calibration dataset.

    Args:
        momentum (float): Momentum factor for smoothing min/max across batches.
            Value between 0 and 1. Higher values give more weight to previous
            statistics. Set to 0 to disable smoothing (strict min/max).
            Default: 0.0 (no smoothing).
    """

    def __init__(self, momentum: float = 0.0):
        """Initialize MinMax calibrator with optional momentum smoothing.

        Args:
            momentum (float): Momentum factor for exponential moving average.
                Must be in range [0, 1). Default: 0.0 (no smoothing).

        Raises:
            ValueError: If momentum is not in valid range.
        """
        super().__init__()
        assert 0 <= momentum < 1, "Momentum must be in the range [0, 1)."

        self.momentum = momentum

    def collect(self, name: str, array: np.ndarray) -> None:
        """Collect min/max statistics from an array.

        Args:
            name (str): Unique identifier for the array
            array (np.ndarray): Activation array to collect statistics from
        """
        current_min = np.min(array)
        current_max = np.max(array)

        if name not in self.data:
            self.data[name] = CalibrationData(min_val=current_min, max_val=current_max)
        else:
            if self.momentum > 0:
                # Smooth updates using exponential moving average
                self.data[name].min_val = (
                    self.momentum * self.data[name].min_val + (1 - self.momentum) * current_min
                )
                self.data[name].max_val = (
                    self.momentum * self.data[name].max_val + (1 - self.momentum) * current_max
                )
            else:
                # No smoothing: strict min/max tracking
                self.data[name].min_val = np.minimum(self.data[name].min_val, current_min)
                self.data[name].max_val = np.maximum(self.data[name].max_val, current_max)

    def compute_range(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute the quantization range.

        Args:
            name (str): Unique identifier for the array

        Returns:
            tuple: (min_val, max_val) for quantization

        Raises:
            KeyError: If no data has been collected for this array
        """
        if name not in self.data:
            raise KeyError(f"No calibration data collected for '{name}'")

        data = self.data[name]

        # Include Zero in the range to have a valid zero point
        min_val = np.minimum(data.min_val, 0)
        max_val = np.maximum(data.max_val, 0)

        return np.array(min_val, dtype=np.float32), np.array(max_val, dtype=np.float32)

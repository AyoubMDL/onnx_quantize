__all__ = ["get_calibrator"]

from typing import Any

from onnx_quantize.core._calibration.base import CalibrationMethod, Calibrator
from onnx_quantize.core._calibration.minmax import MinMaxCalibrator


# Mapping calibration methods to calibrator classes
_CALIBRATORS: dict[CalibrationMethod, type[Calibrator]] = {
    CalibrationMethod.MINMAX: MinMaxCalibrator,
}


def get_calibrator(
    method: CalibrationMethod = CalibrationMethod.MINMAX, **kwargs: Any
) -> Calibrator:
    """Factory function to create a calibrator instance.

    Args:
        method (CalibrationMethod, optional): The calibration method to use. Defaults to MINMAX.
        **kwargs: Additional arguments to pass to the calibrator constructor.

    Returns:
        A calibrator instance
    """
    calibrator_class = _CALIBRATORS[method]

    try:
        return calibrator_class(**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for {calibrator_class.__name__}: {e}") from e

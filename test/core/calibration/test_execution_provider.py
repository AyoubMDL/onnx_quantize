import pytest
from pydantic import ValidationError

from onnx_quantize import CalibrationParams
from onnx_quantize.core._calibration import calibrate as calibrate_mod
from onnx_quantize.core._calibration.base import ExecutionProvider


class TestFromAlias:
    @pytest.mark.parametrize(
        "alias, expected",
        [
            ("cpu", ExecutionProvider.CPU),
            ("CPU", ExecutionProvider.CPU),
            ("Cpu", ExecutionProvider.CPU),
            ("cuda", ExecutionProvider.CUDA),
            ("CUDA", ExecutionProvider.CUDA),
            ("gpu", ExecutionProvider.CUDA),
            ("GPU", ExecutionProvider.CUDA),
        ],
    )
    def test_short_aliases(self, alias, expected):
        assert ExecutionProvider.from_alias(alias) is expected

    @pytest.mark.parametrize(
        "full_name, expected",
        [
            ("CPUExecutionProvider", ExecutionProvider.CPU),
            ("CUDAExecutionProvider", ExecutionProvider.CUDA),
        ],
    )
    def test_full_names(self, full_name, expected):
        assert ExecutionProvider.from_alias(full_name) is expected

    @pytest.mark.parametrize("invalid", ["tpu", "openvino", "", "cudaprovider"])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError, match="Invalid execution provider"):
            ExecutionProvider.from_alias(invalid)


class TestCalibrationParamsProvider:
    def test_default_is_cpu(self):
        params = CalibrationParams()
        assert params.provider is ExecutionProvider.CPU

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("cpu", ExecutionProvider.CPU),
            ("cuda", ExecutionProvider.CUDA),
            ("gpu", ExecutionProvider.CUDA),
            ("CPUExecutionProvider", ExecutionProvider.CPU),
            (ExecutionProvider.CUDA, ExecutionProvider.CUDA),
        ],
    )
    def test_accepts_aliases_and_enum(self, value, expected):
        params = CalibrationParams(provider=value)
        assert params.provider is expected

    def test_rejects_invalid(self):
        with pytest.raises(ValidationError):
            CalibrationParams(provider="tpu")


class TestRequireOnnxruntime:
    def test_raises_with_install_hint_when_missing(self, monkeypatch):
        monkeypatch.setattr(calibrate_mod, "onnxruntime", None)
        with pytest.raises(ImportError, match=r"onnx_quantize\[cpu\].*onnx_quantize\[gpu\]"):
            calibrate_mod._require_onnxruntime()

    def test_passes_when_installed(self):
        # onnxruntime is installed in the test env; should be a no-op
        calibrate_mod._require_onnxruntime()

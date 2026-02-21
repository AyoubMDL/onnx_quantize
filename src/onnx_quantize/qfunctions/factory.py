__all__ = ["get_qfunction"]

from onnx_quantize.core._qconfig import QConfig, QFormat
from onnx_quantize.qfunctions._qdq.qgemm import qgemm_qdq_factory
from onnx_quantize.qfunctions._qdq.qmatmul import qmatmul_qdq_factory
from onnx_quantize.qfunctions._qlinear.qgemm import QLinearGemm
from onnx_quantize.qfunctions._qlinear.qmatmul import QLinearMatMul


_QFUNCTIONS = {
    QFormat.QDQ: {
        "Gemm": qgemm_qdq_factory,
        "MatMul": qmatmul_qdq_factory,
    },
    QFormat.QLINEAR: {
        "Gemm": lambda _: QLinearGemm,
        "MatMul": lambda _: QLinearMatMul,
    },
}


def get_qfunction(op_type: str, qconfig: QConfig):
    """Factory function to get the appropriate quantized operator based on op_type and qconfig.

    Args:
        op_type (str): The type of the operator to be quantized (e.g., "MatMul", "Gemm").
        qconfig (QConfig): The quantization configuration.

    Returns:
        The quantized operator function
    """
    format = qconfig.format
    assert isinstance(format, QFormat)

    return _QFUNCTIONS.get(format, {}).get(op_type, None)(qconfig)

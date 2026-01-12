__all__ = ["get_qfunction"]

from onnx_quantize.core._qconfig import QConfig, QFormat
from onnx_quantize.qfunctions._qdq.qgemm import qgemm_qdq_factory
from onnx_quantize.qfunctions._qdq.qmatmul import qmatmul_qdq_factory
from onnx_quantize.qfunctions._qlinear.qgemm import QLinearGemm
from onnx_quantize.qfunctions._qlinear.qmatmul import QLinearMatMul


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

    if format == QFormat.QDQ:
        if op_type == "MatMul":
            return qmatmul_qdq_factory(qconfig)
        elif op_type == "Gemm":
            return qgemm_qdq_factory(qconfig)
    else:
        if op_type == "MatMul":
            return QLinearMatMul
        elif op_type == "Gemm":
            return QLinearGemm

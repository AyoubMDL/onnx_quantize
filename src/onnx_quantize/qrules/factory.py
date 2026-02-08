__all__ = ["get_qrules"]


import onnxscript

from onnx_quantize.core._qconfig import QConfig, QFormat
from onnx_quantize.qrules._qdq.gemm_to_qgemm import gemm_to_qdq_gemm_rules
from onnx_quantize.qrules._qdq.matmul_to_qmatmul import matmul_to_qdq_matmul_rules
from onnx_quantize.qrules._qlinear.gemm_to_qgemm import gemm_to_qlinear_gemm_rules
from onnx_quantize.qrules._qlinear.matmul_to_qmatmul import matmul_to_qlinear_matmul_rules


_QRULES = {
    QFormat.QDQ: {
        "Gemm": gemm_to_qdq_gemm_rules,
        "MatMul": matmul_to_qdq_matmul_rules,
    },
    QFormat.QLINEAR: {
        "Gemm": gemm_to_qlinear_gemm_rules,
        "MatMul": matmul_to_qlinear_matmul_rules,
    },
}


def get_qrules(qconfig: QConfig) -> list[onnxscript.rewriter.RewriteRuleClassBase]:
    """Returns the list of quantization rules based on the specified quantization format.

    Args:
        qconfig (QConfig): The quantization configuration.

    Returns:
        List of quantization rules applicable for the specified format.
    """
    qrules = []

    for op_type in qconfig.target_op_types:
        qrules.extend(_QRULES.get(qconfig.format, {}).get(op_type, []))

    return qrules

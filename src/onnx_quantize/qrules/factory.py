__all__ = ["get_qrules"]

import onnxscript

from onnx_quantize.core._qconfig import QFormat
from onnx_quantize.qrules._qdq.gemm_to_qgemm import gemm_to_qdq_gemm_rules
from onnx_quantize.qrules._qdq.matmul_to_qmatmul import matmul_to_qdq_matmul_rules
from onnx_quantize.qrules._qlinear.gemm_to_qgemm import gemm_to_qlinear_gemm_rules
from onnx_quantize.qrules._qlinear.matmul_to_qmatmul import matmul_to_qlinear_matmul_rules


def get_qrules(format: QFormat) -> list[onnxscript.rewriter.RewriteRuleClassBase]:
    """Returns the list of quantization rules based on the specified quantization format.

    Args:
        format (QFormat): The quantization format (QDQ or QLINEAR).

    Returns:
        List of quantization rules applicable for the specified format.
    """
    if format == QFormat.QDQ:
        return [*gemm_to_qdq_gemm_rules, *matmul_to_qdq_matmul_rules]

    return [*gemm_to_qlinear_gemm_rules, *matmul_to_qlinear_matmul_rules]

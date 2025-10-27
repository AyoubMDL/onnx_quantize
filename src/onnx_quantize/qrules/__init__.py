from onnx_quantize.qrules.gemm_to_qgemm import gemm_to_qgemm_rules
from onnx_quantize.qrules.matmul_to_qmatmul import matmul_to_qmatmul_rules


qrules = [*gemm_to_qgemm_rules, *matmul_to_qmatmul_rules]

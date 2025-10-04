from onnx_quantize.qrules.matmul_to_qmatmul import matmul_to_qmatmul_rules


qrules = [*matmul_to_qmatmul_rules]

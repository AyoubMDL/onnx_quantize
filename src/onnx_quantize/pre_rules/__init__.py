from onnx_quantize.pre_rules.standarize_gemm import standarize_gemm_rules


pre_rules = [*standarize_gemm_rules]

from onnxscript.rewriter import RewriteRuleSet
from onnxscript.rewriter.rules.common import matmul_add_to_gemm_rule

from onnx_quantize.pre_rules.standarize_gemm import standarize_gemm_rules


pre_rules = RewriteRuleSet([matmul_add_to_gemm_rule, *standarize_gemm_rules])

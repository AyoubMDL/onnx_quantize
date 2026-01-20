import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnxscript
from onnxscript.rewriter.rules.common import matmul_add_to_gemm_rule

from onnx_quantize.pre_rules.standarize_gemm import standarize_gemm_rules


# TODO: rename to passes
pre_rules = ir.passes.Sequential(
    # TODO: maybe add custom naming
    common_passes.NameFixPass(),
    onnxscript.rewriter.RewritePass([matmul_add_to_gemm_rule, *standarize_gemm_rules]),
)

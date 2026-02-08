import logging

import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnxscript
from onnxscript.rewriter.rules.common import matmul_add_to_gemm_rule

from onnx_quantize.core._calibration.calibrate import calibrate_model, get_target_nodes
from onnx_quantize.core._qconfig import GPTQConfig, QConfig, SmoothQuantConfig
from onnx_quantize.pre_passes.smooth_quant import SmoothQuantPass
from onnx_quantize.pre_passes.standarize_gemm import standarize_gemm_rules


logger = logging.getLogger(__name__)


def _add_qconfig_to_nodes(ir_model: ir.Model, qconfig: QConfig) -> None:
    nodes = get_target_nodes(ir_model, qconfig.target_op_types)

    for node in ir_model.graph:
        if node in nodes:
            # Store the qconfig in the node metadata
            node.meta["qconfig"] = qconfig.model_dump()


def _needs_calibration(qconfig: QConfig) -> bool:
    if qconfig.input_activations and qconfig.input_activations.is_static:
        return True

    if qconfig.output_activations and qconfig.output_activations.is_static:
        return True

    if any(isinstance(pre, SmoothQuantConfig) for pre in qconfig.preprocessors):
        return True

    if qconfig.weights and isinstance(qconfig.weights.algorithm, GPTQConfig):
        return True

    return False


def apply_pre_passes(model: ir.Model, qconfig: QConfig) -> ir.Model:
    """Get the pre-processing rules based on the provided preprocessors.

    Args:
        model (ir.Model): The ONNX IR model to be processed.
        qconfig (QConfig): The quantization configuration containing preprocessors.

    Returns:
        ir.Model: The processed ONNX IR model after applying pre-processing rules.
    """
    standard_passes = ir.passes.Sequential(
        # TODO: maybe add custom naming
        common_passes.NameFixPass(),
        onnxscript.rewriter.RewritePass([matmul_add_to_gemm_rule, *standarize_gemm_rules]),
    )

    model = standard_passes(model).model

    # Calibrate the model to compute quantization parameters
    # CHECK: this violates input scales when smooth quant is applied
    if _needs_calibration(qconfig):
        logger.info("Calibrating the model...")
        calibrate_model(model, qconfig)

    _add_qconfig_to_nodes(model, qconfig)

    pre_quantization_passes = []
    for preprocessor in qconfig.preprocessors:
        if isinstance(preprocessor, SmoothQuantConfig):
            pre_quantization_passes.append(
                SmoothQuantPass(alpha=preprocessor.alpha, target_op_types=qconfig.target_op_types)
            )

    pre_quantization_passes.append(common_passes.CheckerPass(full_check=True))
    pre_quantization_passes = ir.passes.Sequential(*pre_quantization_passes)

    return pre_quantization_passes(model).model

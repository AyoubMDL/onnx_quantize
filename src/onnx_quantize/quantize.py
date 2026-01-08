__all__ = ["quantize"]

import logging

import onnx
import onnx_ir as ir
import onnx_ir.passes.common as ir_passes
import onnxscript

from onnx_quantize import OP_TYPES_TO_QUANTIZE
from onnx_quantize.core._calibration.calibrate import calibrate_model, get_target_nodes
from onnx_quantize.core._qconfig import GPTQConfig, QConfig
from onnx_quantize.opset import op
from onnx_quantize.pre_rules import pre_rules
from onnx_quantize.qfunctions import get_qfunctions
from onnx_quantize.qrules import qrules


logger = logging.getLogger(__name__)


def _add_qconfig_to_nodes(ir_model: ir.Model, qconfig: QConfig) -> None:
    nodes = get_target_nodes(ir_model, OP_TYPES_TO_QUANTIZE)

    for node in ir_model.graph:
        if node in nodes:
            # Store the qconfig in the node metadata
            node.meta["qconfig"] = qconfig.model_dump()


def _needs_calibration(qconfig: QConfig) -> bool:
    if qconfig.input_activations and qconfig.input_activations.is_static:
        return True

    if qconfig.output_activations and qconfig.output_activations.is_static:
        return True

    if qconfig.weights and isinstance(qconfig.weights.algorithm, GPTQConfig):
        return True

    return False


def _no_quantization_needed(qconfig: QConfig) -> bool:
    return (
        qconfig.weights is None
        and qconfig.input_activations is None
        and qconfig.output_activations is None
    )


def quantize(model: onnx.ModelProto | ir.Model, qconfig: QConfig) -> onnx.ModelProto | ir.Model:
    """Quantizes an ONNX model using calibration data.

    Args:
        model (onnx.ModelProto | ir.Model): The ONNX model to be quantized
        qconfig (QConfig): Configuration for quantization parameters.

    Returns:
        onnx.ModelProto | ir.Model: The quantized ONNX model.
    """
    if not isinstance(model, (onnx.ModelProto, ir.Model)):
        raise TypeError(
            f"model must be an instance of onnx.ModelProto or onnx_ir.Model, got {type(model)}"
        )

    if _no_quantization_needed(qconfig):
        logger.info("No quantization parameters specified in qconfig. Returning original model.")
        return model

    # Convert to IR model
    is_proto = isinstance(model, onnx.ModelProto)
    if is_proto:
        model = ir.from_proto(model)

    # Optimize model before quantization
    model = onnxscript.optimizer.optimize(model)

    # Run pre rules quant
    logger.info("Applying pre-quantization rules...")
    model = onnxscript.rewriter.rewrite(model, pre_rules)

    # Calibrate the model to compute quantization parameters
    if _needs_calibration(qconfig):
        logger.info("Calibrating the model...")
        calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

    _add_qconfig_to_nodes(model, qconfig)

    # Apply quantization rules to rewrite the model
    logger.info("Applying quantization rules...")
    model = onnxscript.rewriter.rewrite(model, qrules)

    # Update opset version
    onnxscript.version_converter.convert_version(model, target_version=op.version)

    # Add quantization functions to the model
    model.functions.update(get_qfunctions())

    # Remove unused functions
    ir_passes.RemoveUnusedFunctionsPass()(model)

    if is_proto:
        model = ir.to_proto(model)

    return model

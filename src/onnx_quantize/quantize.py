__all__ = ["quantize"]

import logging

import onnx
import onnx_ir as ir
import onnx_ir.passes.common as ir_passes
import onnxscript

from onnx_quantize import OP_TYPES_TO_QUANTIZE, GPTQConfig, QConfig
from onnx_quantize.core import calibrate_model, get_nodes_to_quantize
from onnx_quantize.opset import op
from onnx_quantize.pre_rules import pre_rules
from onnx_quantize.qfunctions import get_qfunctions
from onnx_quantize.qrules import qrules


logger = logging.getLogger(__name__)


def _add_qconfig_to_nodes(ir_model, qconfig):
    nodes = get_nodes_to_quantize(ir_model, OP_TYPES_TO_QUANTIZE)

    for node in ir_model.graph:
        if node in nodes:
            # Store the qconfig in the node metadata
            node.meta["qconfig"] = qconfig.model_dump()


def quantize(model: onnx.ModelProto | ir.Model, qconfig: QConfig) -> onnx.ModelProto | ir.Model:
    """Quantizes an ONNX model using calibration data.

    Args:
        model (onnx.ModelProto | ir.Model): The ONNX model to be quantized
        qconfig (QConfig): Configuration for quantization parameters.

    Returns:
        onnx.ModelProto | ir.Model: The quantized ONNX model.
    """
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
    if (qconfig.is_static and not qconfig.weights_only) or isinstance(
        qconfig.algorithm, GPTQConfig
    ):
        logger.info("Calibrating the model...")
        model = calibrate_model(model, qconfig, OP_TYPES_TO_QUANTIZE)

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

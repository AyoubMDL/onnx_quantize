__all__ = ["quantize"]

import onnx
import onnx_ir as ir
import onnxscript

from onnx_quantize import OP_TYPES_TO_QUANTIZE, GPTQConfig, QConfig
from onnx_quantize.core import calibrate_model, get_nodes_to_quantize
from onnx_quantize.opset import op
from onnx_quantize.pre_rules import pre_rules
from onnx_quantize.qfunctions import get_qfunctions
from onnx_quantize.qrules import qrules


def _add_qconfig_to_nodes(ir_model, qconfig):
    nodes = get_nodes_to_quantize(ir_model, OP_TYPES_TO_QUANTIZE)

    for node in ir_model.graph:
        if node in nodes:
            # Store the qconfig in the node metadata
            node.meta["qconfig"] = qconfig.model_dump()


def quantize(model: onnx.ModelProto, qconfig: QConfig) -> onnx.ModelProto:
    """Quantizes an ONNX model using calibration data.

    Args:
        model (onnx.ModelProto): The ONNX model to be quantized
        qconfig (QConfig): Configuration for quantization parameters.

    Returns:
        onnx.ModelProto: The quantized ONNX model.
    """
    # Convert to IR model
    ir_model = ir.from_proto(model)

    # Optimize model before quantization
    ir_model = onnxscript.optimizer.optimize(ir_model)

    # Run pre rules quant
    ir_model = onnxscript.rewriter.rewrite(ir_model, pre_rules)

    # Calibrate the model to compute quantization parameters
    if (qconfig.is_static and not qconfig.weights_only) or isinstance(
        qconfig.algorithm, GPTQConfig
    ):
        ir_model = calibrate_model(ir_model, qconfig, OP_TYPES_TO_QUANTIZE)

    _add_qconfig_to_nodes(ir_model, qconfig)

    # Apply quantization rules to rewrite the model
    ir_model = onnxscript.rewriter.rewrite(ir_model, qrules)

    # Update opset version
    onnxscript.version_converter.convert_version(ir_model, target_version=op.version)

    # Add quantization functions to the model
    ir_model.functions.update(get_qfunctions())

    return ir.to_proto(ir_model)

__all__ = ["quantize"]

import logging

import onnx
import onnx_ir as ir
import onnx_ir.passes.common as ir_passes
import onnxscript

from onnx_quantize.core._qconfig import QConfig
from onnx_quantize.opset import op
from onnx_quantize.pre_passes import apply_pre_passes
from onnx_quantize.qfunctions import get_qfunctions
from onnx_quantize.qrules import get_qrules


logger = logging.getLogger(__name__)


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
    model = apply_pre_passes(model, qconfig)

    # Apply quantization rules to rewrite the model
    logger.info("Applying quantization rules...")
    qrules = get_qrules(qconfig.format)
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

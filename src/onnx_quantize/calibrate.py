import numpy as np
import onnx_ir as ir
import onnxruntime

from onnx_quantize.core import QConfig, get_quantization_params


OP_TYPES_TO_QUANTIZE = {"MatMul"}


def get_nodes_to_quantize(ir_model, op_types_to_calibrate):
    """Returns a list of nodes to quantize.

    Args:
        ir_model (ir.Model): The target model.
        op_types_to_calibrate (set or list): Operation types to consider for calibration.

    Returns:
        list: Nodes to quantize.
    """
    nodes = [node for node in ir_model.graph if node.op_type in op_types_to_calibrate]

    # Filter nodes which first input is not a constant (Now it is only MatMul)
    return [node for node in nodes if ir.convenience.get_const_tensor(node.inputs[1]) is not None]


def _augment_model(ir_model, nodes_to_calibrate):
    # Add outputs to calibrate to graph outputs
    inputs_to_calibre = []
    for node in nodes_to_calibrate:
        if node.inputs[0].name not in inputs_to_calibre:
            inputs_to_calibre.append(node.inputs[0].name)
            ir_model.graph.outputs.extend([node.inputs[0]])

    return ir_model, inputs_to_calibre


def calibrate_model(ir_model: ir.Model, qconfig: QConfig):
    """Calibrates the model by computing scales and zero-points for specified nodes.

    Args:
        ir_model (ir.Model): The ONNX IR model to be calibrated.
        qconfig (QConfig): Configuration for quantization parameters.

    Returns:
        ir.Model: The calibrated ONNX IR model with scales and zero-points added as metadata
    """
    # Clone model to not change original
    ir_clone = ir.from_proto(ir.to_proto(ir_model))
    nodes_to_calibrate = get_nodes_to_quantize(ir_clone, OP_TYPES_TO_QUANTIZE)

    # Augment graph
    ir_clone, inputs_to_calibre = _augment_model(ir_clone, nodes_to_calibrate)

    clone = ir.to_proto(ir_clone)
    session = onnxruntime.InferenceSession(clone.SerializeToString())

    # Get Calibration Data or generate random data if necessary
    calibration_data = qconfig.calibration_data
    if qconfig.is_static and qconfig.calibration_data is None:
        calibration_data = np.random.randn(
            *[d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
        ).astype(np.float32)

    # Run inference
    inputs_dict = {session.get_inputs()[0].name: calibration_data}
    ort_outs = session.run(inputs_to_calibre, inputs_dict)

    # Construct a dict containing output name and output values
    collected_outputs = {}
    for name, out in zip(inputs_to_calibre, ort_outs, strict=True):
        collected_outputs[name] = out

    for node in ir_model.graph:
        if node.op_type in OP_TYPES_TO_QUANTIZE and node.inputs[0].name in collected_outputs:
            node.meta["input_scale"], node.meta["input_zero_point"] = get_quantization_params(
                collected_outputs[node.inputs[0].name],
                qconfig.activations_dtype,
                qconfig.activations_symmetric,
                per_channel=False,  # We support only per-tensor quantization for activations
            )

    return ir_model

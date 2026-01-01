import logging
from contextlib import contextmanager

import numpy as np
import onnx_ir as ir
import onnxruntime

from onnx_quantize.core._calibration.factory import get_calibrator
from onnx_quantize.core._qconfig import GPTQConfig, QConfig
from onnx_quantize.core._rtn import _compute_qparams


logger = logging.getLogger(__name__)


def get_nodes_to_quantize(ir_model: ir.Model, op_types_to_calibrate: list[str]) -> list[ir.Node]:
    """Returns a list of nodes to quantize.

    Args:
        ir_model (ir.Model): The target model.
        op_types_to_calibrate (set or list): Operation types to consider for calibration.

    Returns:
        list: Nodes to quantize.
    """
    nodes = [node for node in ir_model.graph if node.op_type in op_types_to_calibrate]

    # Filter nodes which second or third input (if it exists) is not a constant
    for node in nodes:
        if ir.convenience.get_const_tensor(node.inputs[1]) is None:
            nodes.remove(node)

        # Bias for Gemm
        if len(node.inputs) > 2 and ir.convenience.get_const_tensor(node.inputs[2]) is None:
            nodes.remove(node)

    return nodes


@contextmanager
def _augment_model(ir_model: ir.Model, nodes_to_calibrate: list[ir.Node]):
    """Context manager that temporarily augments model outputs with node inputs.

    Args:
        ir_model (ir.Model): The target model.
        nodes_to_calibrate (list[ir.Node]): Nodes whose inputs should be added to outputs.

    Yields:
        list: Names of inputs that were added to the outputs.
    """
    # Collect unique input names to calibrate
    inputs_to_calibre = []
    added_outputs = []

    for node in nodes_to_calibrate:
        # TODO: update this when supporting output quantization
        if node.inputs[0].name not in inputs_to_calibre:
            inputs_to_calibre.append(node.inputs[0].name)
            added_outputs.append(node.inputs[0])
            ir_model.graph.outputs.extend([node.inputs[0]])

    try:
        yield inputs_to_calibre

    finally:
        # Remove the added outputs
        for output in added_outputs:
            if output in ir_model.graph.outputs:
                ir_model.graph.outputs.remove(output)


def calibrate_model(ir_model: ir.Model, qconfig: QConfig, op_types_to_calibrate: list) -> ir.Model:
    """Calibrates the model by computing scales and zero-points for specified nodes.

    Args:
        ir_model (ir.Model): The ONNX IR model to be calibrated.
        qconfig (QConfig): Configuration for quantization parameters.
        op_types_to_calibrate (list): List of operation types to calibrate.

    Returns:
        ir.Model: The calibrated ONNX IR model with scales and zero-points added as metadata

    Note:
        Supports mini-batch processing via calibration_params:
        - 'batch_size': Number of samples per batch (default: process all at once)
        - 'num_samples': Total number of samples to use (default: all available)
    """
    # Clone model to not change original
    nodes_to_calibrate = get_nodes_to_quantize(ir_model, op_types_to_calibrate)

    # Augment graph with context manager
    with _augment_model(ir_model, nodes_to_calibrate) as inputs_to_calibre:
        proto = ir.to_proto(ir_model)
        session = onnxruntime.InferenceSession(proto.SerializeToString())

        # Get Calibration Data or generate random data
        calibration_data = qconfig.calibration_data
        if calibration_data is None:
            logger.info("Generating random calibration data as none was provided.")
            calibration_data = np.random.randn(
                *[d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
            ).astype(np.float32)

        # Create calibrator based on configuration
        calibrator_params = qconfig.calibration_params or {}
        calibrator = get_calibrator(qconfig.calibration_method, **calibrator_params)

        # Run inference and collect outputs
        inputs_dict = {session.get_inputs()[0].name: calibration_data}
        ort_outs = session.run(inputs_to_calibre, inputs_dict)

        # Construct a dict containing output name and output arrays
        # This used for gptq as it needs to access the full collected data
        collected_outputs = {}
        for name, out in zip(inputs_to_calibre, ort_outs, strict=True):
            collected_outputs[name] = out
            # Collect calibration data for each array
            calibrator.collect(name, out)

    # Compute quantization parameters for each node
    for node in ir_model.graph:
        if node.op_type in op_types_to_calibrate and node.inputs[0].name in calibrator.data:
            array_name = node.inputs[0].name

            if isinstance(qconfig.algorithm, GPTQConfig):
                # Collect input activations to compute Hessian
                node.meta["input"] = collected_outputs[array_name]
                # TODO: maybe clear the current outputs to save memory
                continue  # GPTQ is weight only quantization

            # Use calibrator to compute the quantization range
            rmin, rmax = calibrator.compute_range(array_name)

            node.meta["input_scale"], node.meta["input_zero_point"] = _compute_qparams(
                rmin,
                rmax,
                qconfig.activations_dtype,
                qconfig.activations_symmetric,
                qconfig.reduce_range,
            )

    return ir_model

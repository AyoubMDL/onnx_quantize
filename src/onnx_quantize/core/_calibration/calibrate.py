import collections
import enum
import logging
import tempfile
from contextlib import contextmanager

import numpy as np
import onnx_ir as ir
import onnxruntime

from onnx_quantize.core._algorithms.utils import _compute_qparams
from onnx_quantize.core._calibration.factory import Calibrator, get_calibrator
from onnx_quantize.core._qconfig import (
    AwqConfig,
    GPTQConfig,
    QActivationArgs,
    QConfig,
    SmoothQuantConfig,
)


logger = logging.getLogger(__name__)


class _ActivationKind(enum.Enum):
    INPUT = "input"
    OUTPUT = "output"


def get_target_nodes(ir_model: ir.Model, op_types_to_calibrate: set[str]) -> set[ir.Node]:
    """Returns a set of nodes to quantize.

    Args:
        ir_model (ir.Model): The ONNX IR model to analyze.
        op_types_to_calibrate (set[str]): Set of operation types to calibrate.

    Returns:
        set[ir.Node]: A set of nodes to quantize.
    """

    def is_valid(node: ir.Node) -> bool:
        if node.op_type not in op_types_to_calibrate:
            return False

        # Weight must be constant
        if ir.convenience.get_const_tensor(node.inputs[1]) is None:
            return False

        # Optional bias must be constant
        if len(node.inputs) > 2 and ir.convenience.get_const_tensor(node.inputs[2]) is None:
            return False

        return True

    return {node for node in ir_model.graph if is_valid(node)}


def _get_values_to_calibrate(
    target_nodes: set[ir.Node], *, augment_inputs: bool, augment_outputs: bool
) -> set[ir.Value]:
    """Collect values (inputs/outputs) to calibrate from target nodes."""
    values = set()

    for node in target_nodes:
        if augment_inputs:
            values.add(node.inputs[0])

        if augment_outputs:
            values.add(node.outputs[0])

    return values


@contextmanager
def _augment_model(ir_model: ir.Model, values: set[ir.Value]):
    """Temporarily augment model outputs to include specified values."""
    original_outputs = list(ir_model.graph.outputs)

    for v in values:
        if v not in ir_model.graph.outputs:
            ir_model.graph.outputs.append(v)

    try:
        yield [v.name for v in ir_model.graph.outputs]

    finally:
        ir_model.graph.outputs[:] = original_outputs


def _generate_random_calibration_data(
    num_samples: int,
    session_inputs,  # list[onnxruntime.NodeArg]
) -> np.ndarray | dict[str, np.ndarray]:
    logger.info("Generating random calibration data as None was provided.")
    rng = np.random.default_rng(0)

    def _random_array(input_shape) -> np.ndarray:
        shape = [num_samples] + [d if isinstance(d, int) else 1 for d in input_shape[1:]]
        return rng.standard_normal(size=shape).astype(np.float32)

    if len(session_inputs) == 1:
        return _random_array(session_inputs[0].shape)

    return {inp.name: _random_array(inp.shape) for inp in session_inputs}


def _prepare_calibration_data(
    calibration_data: np.ndarray, batch_size: int, num_samples: int
) -> np.ndarray:
    """Prepare calibration data for mini-batch processing."""
    total_samples = calibration_data.shape[0]

    if num_samples > total_samples:
        num_samples = total_samples

    calibration_data = calibration_data[:num_samples]

    if batch_size >= num_samples:
        return calibration_data.reshape((1, num_samples, *calibration_data.shape[1:]))

    # Reshape data into batches
    # Drop excess samples that don't fit into a full batch
    num_batches = num_samples // batch_size
    calibration_data = calibration_data[: num_batches * batch_size]
    calibration_data = calibration_data.reshape(
        (num_batches, batch_size, *calibration_data.shape[1:])
    )

    return calibration_data


def _collect_activations(
    ir_model: ir.Model,
    values_to_calibrate: set[ir.Value],
    calibration_data: np.ndarray | dict[str, np.ndarray] | None,
    num_samples: int,
    batch_size: int,
) -> list[dict[str, np.ndarray]]:
    """Collect interm activation values from the model during inference."""
    # Augment model graph outputs to collect required activations
    with (
        _augment_model(ir_model, values_to_calibrate) as values_names,
        tempfile.NamedTemporaryFile() as tmpfile,
    ):
        ir.save(ir_model, tmpfile.name)
        # TODO: specify providers
        session = onnxruntime.InferenceSession(tmpfile.name)

        if calibration_data is None:
            calibration_data = _generate_random_calibration_data(num_samples, session.get_inputs())

        if len(session.get_inputs()) > 1:
            if not isinstance(calibration_data, dict):
                raise ValueError(
                    "Calibration data must be a dict mapping input names to arrays "
                    "for multi-input models."
                )

        if isinstance(calibration_data, np.ndarray):
            calibration_data = {session.get_inputs()[0].name: calibration_data}

        batched_inputs = {
            name: _prepare_calibration_data(data, batch_size, num_samples)
            for name, data in calibration_data.items()
        }

        num_batches = len(next(iter(batched_inputs.values())))
        activations = []

        for i in range(num_batches):
            feed_dict = {name: data[i] for name, data in batched_inputs.items()}
            outputs = session.run(values_names, feed_dict)
            activations.append(dict(zip(values_names, outputs, strict=True)))

    return activations


def _set_qparams(
    ir_model: ir.Model,
    activations: list[dict[str, np.ndarray]],
    nodes_to_calibrate: set[ir.Node],
    calibrator: Calibrator,
    qargs: QActivationArgs,
    kind: _ActivationKind,
) -> None:
    """Compute and set quantization parameters (scale and zero-point) for activations."""
    # Run Calibrator to later compute ranges
    for activation in activations:
        for name, data in activation.items():
            calibrator.collect(name, data)

    for node in ir_model.graph:
        name = node.inputs[0].name if kind == _ActivationKind.INPUT else node.outputs[0].name

        if node in nodes_to_calibrate and name in calibrator.data:
            # Use calibrator to compute the quantization range
            rmin, rmax = calibrator.compute_range(name)

            node.meta[f"{kind.value}_scale"], node.meta[f"{kind.value}_zero_point"] = (
                _compute_qparams(
                    rmin,
                    rmax,
                    qargs.dtype,
                    qargs.symmetric,
                    qargs.reduce_range,
                    qargs.scale_dtype,
                    qargs.zp_dtype,
                )
            )


def _set_qparams_gptq(
    ir_model: ir.Model,
    activations: list[dict[str, np.ndarray]],
    nodes_to_calibrate: set[ir.Node],
) -> None:
    """Collect and store input activations for GPTQ quantization."""
    # As for GPTQ, we just need to collect input activations for each node,
    # therefore, we transform the list of outputs into a dict and concatenate along axis 0
    collected_outputs = collections.defaultdict(list)
    for activation in activations:
        for name, data in activation.items():
            collected_outputs[name].append(data)

    for name in collected_outputs:
        collected_outputs[name] = np.concatenate(collected_outputs[name], axis=0)

    for node in ir_model.graph:
        if node in nodes_to_calibrate and node.inputs[0].name in collected_outputs:
            # Collect input activations to compute Hessian
            node.meta["input"] = collected_outputs[node.inputs[0].name]


def calibrate_model(ir_model: ir.Model, qconfig: QConfig):
    """Calibrates the model by computing scales and zero-points for specified nodes.

    Args:
        ir_model (ir.Model): The ONNX IR model to be calibrated.
        qconfig (QConfig): Configuration for quantization parameters.
    """
    # Get target nodes to calibrate
    nodes_to_calibrate = get_target_nodes(ir_model, qconfig.target_op_types)

    # Identify which activations to calibrate
    calibrate_inputs = qconfig.input_activations is not None and qconfig.input_activations.is_static
    calibrate_outputs = (
        qconfig.output_activations is not None and qconfig.output_activations.is_static
    )
    smooth_quant = any(isinstance(pre, SmoothQuantConfig) for pre in qconfig.preprocessors)
    awq = any(isinstance(pre, AwqConfig) for pre in qconfig.preprocessors)
    gptq = qconfig.weights is not None and isinstance(qconfig.weights.algorithm, GPTQConfig)

    values_to_calibrate = _get_values_to_calibrate(
        nodes_to_calibrate,
        augment_inputs=calibrate_inputs or gptq or smooth_quant or awq,
        augment_outputs=calibrate_outputs,
    )

    # Extract calibration parameters
    calibrator_params = qconfig.calibration_params.model_dump()
    batch_size = calibrator_params.pop("batch_size")
    num_samples = calibrator_params.pop("num_samples")

    activations = _collect_activations(
        ir_model,
        values_to_calibrate=values_to_calibrate,
        calibration_data=qconfig.calibration_data,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    # Create calibrator based on configuration
    method = calibrator_params.pop("method")
    calibrator = get_calibrator(method, **calibrator_params)

    if calibrate_inputs:
        _set_qparams(
            ir_model,
            activations,
            nodes_to_calibrate,
            calibrator,
            qconfig.input_activations,
            _ActivationKind.INPUT,
        )

    if calibrate_outputs:
        _set_qparams(
            ir_model,
            activations,
            nodes_to_calibrate,
            calibrator,
            qconfig.output_activations,
            _ActivationKind.OUTPUT,
        )

    if gptq or smooth_quant or awq:
        # This is special case where we need to collect input activations for GPTQ/SmoothQuant/AWQ
        # No need for calibrator here.
        _set_qparams_gptq(
            ir_model,
            activations,
            nodes_to_calibrate,
        )

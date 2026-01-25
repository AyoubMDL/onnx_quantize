import numpy as np
import onnx
import onnx_ir as ir
import onnxscript
import pytest

from onnx_quantize import GPTQConfig, HqqConfig, QuantType
from onnx_quantize.core._calibration.calibrate import calibrate_model
from onnx_quantize.core._qconfig import QActivationArgs, QConfig, QWeightArgs
from onnx_quantize.opset import op
from onnx_quantize.qfunctions._qdq.qmatmul import qmatmul_qdq_factory
from onnx_quantize.qrules._qdq.matmul_to_qmatmul import matmul_to_qdq_matmul_rules
from onnx_quantize.quantize import _add_qconfig_to_nodes

from ...helpers import onnx_forward_on_models


def _get_test_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = MatMul(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(rng.uniform(size=(32, 64)).astype(np.float32), name="W1")
    W2 = onnx.numpy_helper.from_array(rng.uniform(size=(64, 128)).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, W2])

    model = ir.from_proto(model)
    return model


def _test_matmul_to_qmatmul(rng, model, qconfig):
    _add_qconfig_to_nodes(model, qconfig)
    model = onnxscript.rewriter.rewrite(model, matmul_to_qdq_matmul_rules)

    expected_op = qmatmul_qdq_factory(qconfig)
    for node in model.graph:
        assert node.op_type == expected_op.__name__

    # Convert model to target opset version
    onnxscript.version_converter.convert_version(model, target_version=op.version)

    # Add functions to check inference later
    func = ir.serde.deserialize_function(expected_op.to_function_proto())
    model.functions[func.identifier()] = func

    # Check model
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto)

    # Check that inference runs without error
    samples = rng.uniform(size=(1, 32)).astype(np.float32)
    onnx_forward_on_models(proto, samples={"X": samples})


@pytest.mark.parametrize("strategy", ["tensor", "channel"])
@pytest.mark.parametrize("algo", [None, GPTQConfig()])
def test_matmul_to_qmatmul_weights_only(rng, strategy, algo):
    model = _get_test_model(rng)

    # Create QConfig with new structure
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy=strategy,
            algorithm=algo,
        )
    )

    if isinstance(algo, GPTQConfig):
        calibrate_model(model, qconfig, op_types_to_calibrate={"MatMul"})
    _test_matmul_to_qmatmul(rng, model, qconfig)


@pytest.mark.parametrize("group_size", [16, 128])
@pytest.mark.parametrize("algo", [None, GPTQConfig()])
def test_matmul_to_qmatmul_weights_only_matmul_nbits(rng, group_size, algo):
    model = _get_test_model(rng)

    # Create QConfig with new structure
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="group",
            group_size=group_size,
            algorithm=algo,
        )
    )

    if isinstance(algo, GPTQConfig):
        calibrate_model(model, qconfig, op_types_to_calibrate={"MatMul"})

    _add_qconfig_to_nodes(model, qconfig)
    model = onnxscript.rewriter.rewrite(model, matmul_to_qdq_matmul_rules)

    for node in model.graph:
        assert node.op_type == "MatMulNBits"

    # Check model
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto)

    # Check that inference runs without error
    samples = rng.uniform(size=(1, 32)).astype(np.float32)
    onnx_forward_on_models(proto, samples={"X": samples})


@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_matmul_to_qmatmul_weights_only_hqq(rng, group_size):
    """Test HQQ quantization for MatMul to QMatMul transformation."""
    model = _get_test_model(rng)

    # Create QConfig with HQQ
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy="group",
            group_size=group_size,
            symmetric=False,
            algorithm=HqqConfig(),
        )
    )

    _add_qconfig_to_nodes(model, qconfig)
    model = onnxscript.rewriter.rewrite(model, matmul_to_qdq_matmul_rules)

    for node in model.graph:
        assert node.op_type == "MatMulNBits"

    # Check model
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto)

    # Check that inference runs without error
    samples = rng.uniform(size=(1, 32)).astype(np.float32)
    onnx_forward_on_models(proto, samples={"X": samples})


@pytest.mark.parametrize("is_static", [True, False])
def test_matmul_to_qmatmul_weights_inputs(rng, is_static):
    model = _get_test_model(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="tensor",
        ),
        input_activations=QActivationArgs(
            dtype=QuantType.QUInt8,
            is_static=is_static,
        ),
    )

    if is_static:
        calibrate_model(model, qconfig, op_types_to_calibrate={"MatMul"})
    _test_matmul_to_qmatmul(rng, model, qconfig)


@pytest.mark.parametrize("is_static", [True, False])
def test_matmul_to_qmatmul_weights_outputs(rng, is_static):
    model = _get_test_model(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="tensor",
        ),
        output_activations=QActivationArgs(
            dtype=QuantType.QUInt8,
            is_static=is_static,
        ),
    )

    if is_static:
        calibrate_model(model, qconfig, op_types_to_calibrate={"MatMul"})
    _test_matmul_to_qmatmul(rng, model, qconfig)


@pytest.mark.parametrize("is_static", [True, False])
def test_matmul_to_qmatmul_weights_inputs_outputs(rng, is_static):
    model = _get_test_model(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="tensor",
        ),
        input_activations=QActivationArgs(
            dtype=QuantType.QUInt8,
            is_static=is_static,
        ),
        output_activations=QActivationArgs(
            dtype=QuantType.QUInt8,
            is_static=is_static,
        ),
    )

    if is_static:
        calibrate_model(model, qconfig, op_types_to_calibrate={"MatMul"})
    _test_matmul_to_qmatmul(rng, model, qconfig)

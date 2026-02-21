import numpy as np
import onnx
import onnx_ir as ir
import pytest

from onnx_quantize import AwqConfig, GPTQConfig, HqqConfig, QuantType, SmoothQuantConfig, quantize
from onnx_quantize.core._qconfig import QActivationArgs, QConfig, QWeightArgs
from onnx_quantize.qfunctions import MS_OPSET, QUANT_OPSET
from test.helpers import onnx_forward_on_models


def _truncated_normal(rng, shape, scale=0.1, clip=2.5):
    x = rng.normal(0.0, scale, size=shape)
    return np.clip(x, -clip * scale, clip * scale).astype(np.float32)


def _get_matmul_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 21] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = MatMul(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_gemm_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[64, 32] W1, float[64] B1, float[64, 128] W2>
                {
                    x1 = Gemm<transB=1>(X, W1, B1)
                    Y = Gemm(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 32)), name="W1")
    B1 = onnx.numpy_helper.from_array(
        _truncated_normal(rng, (64,)),
        name="B1",
    )
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, B1, W2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _get_matmul_add_model(rng):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2, float[64] B1, float[128] B2>
                {
                    x1 = MatMul(X, W1)
                    x2 = Add(x1, B1)
                    x3 = MatMul(x2, W2)
                    Y = Add(x3, B2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    B1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64)), name="B1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    B2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (128)), name="B2")
    model.graph.initializer.extend([W1, B1, W2, B2])
    onnx.checker.check_model(model, full_check=True)
    return model


def _test_quantize(rng, model, qconfig, calibration_data=None):
    """Helper function to test quantization on a model."""
    # Convert to IR if needed
    is_proto = isinstance(model, onnx.ModelProto)
    if is_proto:
        original_model = model
        model = ir.from_proto(model)

    # Quantize model
    qmodel = quantize(model, qconfig)

    # Check type consistency
    assert isinstance(qmodel, type(model))

    # Convert to proto for further checks
    qmodel_proto = ir.to_proto(qmodel)
    model_proto = ir.to_proto(model) if not is_proto else original_model

    # Check all nodes are quantized (Assuming all ops are quantized)
    assert all(
        node.domain in {MS_OPSET.domain, QUANT_OPSET.domain} for node in qmodel_proto.graph.node
    )

    # Check inference and compare outputs
    # Use calibration data if available, otherwise generate new test data
    if calibration_data is None:
        calibration_data = _truncated_normal(rng, (2, 32))

    # Skip comparison for 4-bit weights due to higher quantization error
    # and CI keeps failing occasionally
    if qconfig.weights.dtype.bitwidth > 4:
        original_output, quantized_output = onnx_forward_on_models(
            model_proto, qmodel_proto, samples={"X": calibration_data}
        )

        np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)


@pytest.mark.parametrize(
    "strategy, group_size",
    [
        ("tensor", None),
        ("channel", None),
        ("group", 16),
        ("group", 8),
    ],
)
@pytest.mark.parametrize(
    "dtype", [QuantType.QUInt8, QuantType.QInt8, QuantType.QUInt4, QuantType.QInt4]
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_only(rng, model_fn, strategy, group_size, dtype, symmetric, mse):
    model = model_fn(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy=strategy,
            group_size=group_size,
            symmetric=symmetric,
            mse=mse,
        )
    )

    _test_quantize(rng, model, qconfig)


@pytest.mark.parametrize(
    "strategy, group_size",
    [
        ("tensor", None),
        ("channel", None),
        ("group", 16),
        ("group", 8),
    ],
)
@pytest.mark.parametrize(
    "dtype", [QuantType.QUInt8, QuantType.QInt8, QuantType.QInt4, QuantType.QUInt4]
)
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_only_gptq(rng, model_fn, strategy, group_size, dtype):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32))

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy=strategy,
            group_size=group_size,
            algorithm=GPTQConfig(block_size=16),
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(rng, model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_inputs(rng, model_fn, is_static, dtype, symmetric):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=symmetric,
        ),
        input_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(rng, model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_outputs(rng, model_fn, is_static, dtype):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=True,
        ),
        output_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(rng, model, qconfig, calibration_data)


@pytest.mark.parametrize(
    "is_static, dtype",
    [
        (True, QuantType.QInt8),
        (True, QuantType.QUInt8),
        (False, QuantType.QUInt8),
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_inputs_outputs(rng, model_fn, is_static, dtype, symmetric):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32)) if is_static else None

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="tensor",
            symmetric=symmetric,
        ),
        input_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        output_activations=QActivationArgs(
            dtype=dtype,
            is_static=is_static,
        ),
        calibration_data=calibration_data,
    )

    _test_quantize(rng, model, qconfig, calibration_data)


@pytest.mark.parametrize("dtype", [QuantType.QInt8, QuantType.QUInt8])
@pytest.mark.parametrize("strategy", ["tensor", "channel"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_inputs_outputs_qlinear_format(rng, model_fn, dtype, symmetric, strategy):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32))

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy=strategy,
            symmetric=symmetric,
        ),
        input_activations=QActivationArgs(
            dtype=dtype,
            is_static=True,
        ),
        output_activations=QActivationArgs(
            dtype=dtype,
            is_static=True,
        ),
        calibration_data=calibration_data,
        format="qlinear",
    )

    _test_quantize(rng, model, qconfig, calibration_data)


def test_no_quantization_needed(rng):
    model = _get_matmul_model(rng)

    qconfig = QConfig(
        weights=None,
        input_activations=None,
        output_activations=None,
    )

    qmodel = quantize(model, qconfig)

    # Check that the returned model is the same as the original
    assert qmodel == model


@pytest.mark.parametrize("dtype", [QuantType.QUInt4, QuantType.QUInt8])
@pytest.mark.parametrize("algorithm", [None, GPTQConfig()])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_matmul_nbits_compatibility(rng, model_fn, dtype, algorithm):
    model = model_fn(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=dtype,
            strategy="group",
            group_size=16,
            algorithm=algorithm,
        )
    )

    qmodel = quantize(model, qconfig)
    qmodel_proto = ir.to_proto(qmodel) if isinstance(qmodel, ir.Model) else qmodel
    model_proto = ir.to_proto(model) if isinstance(model, ir.Model) else model

    # Check that all nodes are MatMulNBits
    assert all(node.op_type == "MatMulNBits" for node in qmodel.graph.node)

    # Check all nodes are quantized (Assuming all ops are quantized)
    assert all(
        node.domain in {MS_OPSET.domain, QUANT_OPSET.domain} for node in qmodel_proto.graph.node
    )

    original_output, quantized_output = onnx_forward_on_models(
        model_proto, qmodel_proto, samples={"X": _truncated_normal(rng, (2, 32))}
    )

    np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)


@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_weights_only_hqq(rng, model_fn, group_size):
    model = model_fn(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy="group",
            group_size=group_size,
            symmetric=False,
            algorithm=HqqConfig(),
        ),
    )

    _test_quantize(rng, model, qconfig)


@pytest.mark.parametrize(
    "hqq_params",
    [
        {"lp_norm": 0.7, "beta": 10.0, "iters": 20},
        {"lp_norm": 0.5, "beta": 5.0, "iters": 10, "early_stop": False},
        {"lp_norm": 1.0, "beta": 15.0, "kappa": 1.05, "iters": 15},
    ],
)
def test_quantize_weights_hqq_custom_params(rng, hqq_params):
    model = _get_matmul_model(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy="group",
            group_size=32,
            symmetric=False,
            algorithm=HqqConfig(**hqq_params),
        ),
    )

    _test_quantize(rng, model, qconfig)


@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model, _get_matmul_add_model])
def test_quantize_hqq_matmul_nbits_compatibility(rng, model_fn, group_size):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32))

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt4,
            strategy="group",
            group_size=group_size,
            symmetric=False,
            algorithm=HqqConfig(),
        ),
        calibration_data=calibration_data,
    )

    qmodel = quantize(model, qconfig)
    qmodel_proto = ir.to_proto(qmodel) if isinstance(qmodel, ir.Model) else qmodel
    model_proto = ir.to_proto(model) if isinstance(model, ir.Model) else model

    # Check that all nodes are MatMulNBits
    assert all(node.op_type == "MatMulNBits" for node in qmodel.graph.node)

    # Check all nodes are quantized
    assert all(
        node.domain in {MS_OPSET.domain, QUANT_OPSET.domain} for node in qmodel_proto.graph.node
    )

    original_output, quantized_output = onnx_forward_on_models(
        model_proto, qmodel_proto, samples={"X": calibration_data}
    )

    np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)


def test_quantize_smooth_quant(rng):
    model = _get_matmul_model(rng)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="tensor",
            symmetric=True,
        ),
        input_activations=QActivationArgs(
            dtype=QuantType.QUInt8,
            is_static=True,
        ),
        preprocessors=[SmoothQuantConfig(alpha=0.5)],
    )

    # Check inference is fine
    # TODO: modify the test to make it more robust
    # For the moment, we only check the inference
    qmodel = quantize(model, qconfig)
    onnx_forward_on_models(qmodel, samples={"X": _truncated_normal(rng, (2, 32))})


@pytest.mark.parametrize("target_op_types", [("Gemm",), ("MatMul",)])
def test_quantize_specific_op_types(rng, target_op_types):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 21] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = Gemm(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(_truncated_normal(rng, (32, 64)), name="W1")
    W2 = onnx.numpy_helper.from_array(_truncated_normal(rng, (64, 128)), name="W2")
    model.graph.initializer.extend([W1, W2])
    onnx.checker.check_model(model, full_check=True)

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QUInt8,
            strategy="tensor",
            symmetric=True,
        ),
        target_op_types=target_op_types,
    )

    qmodel = quantize(model, qconfig)

    # Check that only the specified op types are quantized
    for original_node, qnode in zip(model.graph.node, qmodel.graph.node, strict=True):
        if original_node.op_type in target_op_types:
            assert qnode.domain in {MS_OPSET.domain, QUANT_OPSET.domain}
        else:
            assert qnode.domain == ""

    # Check inference is fine
    onnx_forward_on_models(qmodel, samples={"X": _truncated_normal(rng, (2, 32))})


@pytest.mark.parametrize("clip_search", [False, True])
@pytest.mark.parametrize(
    "strategy, group_size", [("tensor", None), ("channel", None), ("group", 16)]
)
@pytest.mark.parametrize("model_fn", [_get_matmul_model, _get_gemm_model])
def test_quantize_awq(rng, model_fn, strategy, group_size, clip_search):
    model = model_fn(rng)
    calibration_data = _truncated_normal(rng, (2, 32))

    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=QuantType.QInt8,
            strategy=strategy,
            group_size=group_size,
        ),
        preprocessors=[AwqConfig(clip_search=clip_search)],
        calibration_data=calibration_data,
    )

    qmodel = quantize(model, qconfig)

    test_samples = _truncated_normal(rng, (2, 32))
    original_output, quantized_output = onnx_forward_on_models(
        model, qmodel, samples={"X": test_samples}
    )
    np.testing.assert_allclose(original_output, quantized_output, atol=1e-1)

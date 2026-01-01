import numpy as np
import onnx
import onnx_ir as ir
import onnxscript
import pytest

from onnx_quantize import QConfig, QuantType
from onnx_quantize.core._calibration.calibrate import calibrate_model
from onnx_quantize.qrules.matmul_to_qmatmul import matmul_to_qmatmul_rules
from onnx_quantize.quantize import _add_qconfig_to_nodes


@pytest.mark.parametrize("is_static", [True, False])
@pytest.mark.parametrize(
    "weights_only, strategy, group_size",
    [
        (True, "tensor", None),
        (True, "tensor", None),
        (True, "channel", None),
        (True, "channel", None),
        (True, "group", 16),
        (True, "group", 8),
    ],
)
def test_matmul_to_qmatmul(is_static, weights_only, strategy, group_size):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64, 128] W2>
                {
                    x1 = MatMul(X, W1)
                    Y = MatMul(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(32, 64).astype(np.float32), name="W1")
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, W2])

    model = ir.from_proto(model)
    qconfig = QConfig(
        is_static=is_static,
        weights_only=weights_only,
        weights_dtype=QuantType.QUInt8,
        strategy=strategy,
        group_size=group_size,
    )

    if is_static and not weights_only:
        model = calibrate_model(model, qconfig)

    _add_qconfig_to_nodes(model, qconfig)
    model = onnxscript.rewriter.rewrite(model, matmul_to_qmatmul_rules)

    if weights_only:
        # Check that all nodes are of type 'QMatMulWeightsOnly'
        for node in model.graph:
            expected_op_type = "QMatMulWeightsOnly"
            if strategy == "group":
                expected_op_type += "Grouped"

            assert node.op_type == expected_op_type

    elif is_static:
        # Check that all nodes are of type 'QMatMulStatic8bits'
        for node in model.graph:
            assert node.op_type == "QMatMulStatic8bits"
    else:
        # Check that all nodes are of type 'QMatMulDynamic8bits'
        for node in model.graph:
            assert node.op_type == "QMatMulDynamic8bits"

    # Check model
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto)

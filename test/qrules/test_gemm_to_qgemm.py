import numpy as np
import onnx
import onnx_ir as ir
import onnxscript
import pytest

from onnx_quantize import QConfig, QuantType
from onnx_quantize.core._calibrate import calibrate_model
from onnx_quantize.qrules.gemm_to_qgemm import gemm_to_qgemm_rules
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
def test_gemm_to_qgemm(is_static, weights_only, strategy, group_size):
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[32, 64] W1, float[64] B1, float[64, 128] W2>
                {
                    x1 = Gemm<transB=0>(X, W1, B1)
                    Y = Gemm<transB=0>(x1, W2)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(32, 64).astype(np.float32), name="W1")
    B1 = onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), name="B1")
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    model.graph.initializer.extend([W1, B1, W2])

    model = ir.from_proto(model)
    clone = ir.from_proto(ir.to_proto(model))

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
    model = onnxscript.rewriter.rewrite(model, gemm_to_qgemm_rules)

    if weights_only:
        for org_node, node in zip(clone.graph, model.graph, strict=True):
            # Check for bias
            if len(org_node.inputs) == 3:
                expected_op_type = "QGemmWeightsOnly"
            else:
                expected_op_type = "QMatMulWeightsOnly"

            if strategy == "group":
                expected_op_type += "Grouped"

            assert node.op_type == expected_op_type
    elif is_static:
        for org_node, node in zip(clone.graph, model.graph, strict=True):
            # Check for bias
            if len(org_node.inputs) == 3:
                assert node.op_type == "QGemmStatic8bits"
            else:
                assert node.op_type == "QMatMulStatic8bits"
    else:
        # Check that all nodes are of type 'QMatMulDynamic8bits'
        for org_node, node in zip(clone.graph, model.graph, strict=True):
            # Check for bias
            if len(org_node.inputs) == 3:
                assert node.op_type == "QGemmDynamic8bits"
            else:
                assert node.op_type == "QMatMulDynamic8bits"

    # Check model
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto)

import numpy as np
import onnx
import onnx_ir as ir
import onnxscript

from onnx_quantize.pre_passes.standarize_gemm import standarize_gemm_rules


def _get_gemm_model():
    model = onnx.parser.parse_model("""
                < ir_version: 10, opset_import: ["" : 20] >
                test_model (float[N, 32] X) => (float [N, ?] Y)
                <float[64, 32] W1, float[64] B1, float[64, 128] W2,
                 float[128, 256] W3>
                {
                    x1 = Gemm<transB=1>(X, W1, B1)
                    x2 = Gemm(x1, W2)
                    Y = Gemm<transB=0>(x2, W3)
                }
            """)
    W1 = onnx.numpy_helper.from_array(np.random.randn(64, 32).astype(np.float32), name="W1")
    B1 = onnx.numpy_helper.from_array(
        np.random.randn(
            64,
        ).astype(np.float32),
        name="B1",
    )
    W2 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W2")
    W3 = onnx.numpy_helper.from_array(np.random.randn(128, 256).astype(np.float32), name="W3")
    model.graph.initializer.extend([W1, B1, W2, W3])
    return ir.from_proto(model)


def test_standarize_gemm():
    model = _get_gemm_model()
    model = onnxscript.rewriter.rewrite(model, standarize_gemm_rules)

    for node in model.graph:
        assert node.attributes.get("transB").value == 0

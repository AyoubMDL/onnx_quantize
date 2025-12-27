from onnx_quantize.opset import op


def test_opset_version():
    assert op.version == 21

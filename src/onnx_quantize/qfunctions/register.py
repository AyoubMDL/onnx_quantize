import onnxscript


_QFUNCTIONS = []
QUANT_OPSET = onnxscript.values.Opset(domain="quant", version=1)


def register_qfunction(func):
    """Decorator to register a quantization function by adding its proto to the global list."""
    _QFUNCTIONS.append(func.to_function_proto())
    return func

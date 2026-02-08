import onnxscript


_QFUNCTIONS = []
QUANT_OPSET = onnxscript.values.Opset(domain="quant", version=1)
MS_OPSET = onnxscript.values.Opset("com.microsoft", version=1)


def register_qfunction(_func=None):
    """Decorator to register a quantization function by adding its proto to the global list."""

    def wrapper(func):
        _QFUNCTIONS.append(func.to_function_proto())
        return func

    if _func is None:
        return wrapper
    return wrapper(_func)

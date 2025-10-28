import onnxscript


_QFUNCTIONS = []
OP_TYPES_TO_QUANTIZE = set()
QUANT_OPSET = onnxscript.values.Opset(domain="quant", version=1)


def register_qfunction(_func=None, *, target_optype=None):
    """Decorator to register a quantization function by adding its proto to the global list."""

    def wrapper(func):
        _QFUNCTIONS.append(func.to_function_proto())
        if target_optype:
            OP_TYPES_TO_QUANTIZE.add(target_optype)
        return func

    if _func is None:
        return wrapper
    return wrapper(_func)

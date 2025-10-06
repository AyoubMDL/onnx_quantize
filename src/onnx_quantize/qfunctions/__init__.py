import onnx_ir as ir

from onnx_quantize.qfunctions.qmatmul import *
from onnx_quantize.qfunctions.register import _QFUNCTIONS, register_qfunction


def _get_functions():
    functions = {}
    for func in _QFUNCTIONS:
        func = ir.serde.deserialize_function(func)
        functions[func.identifier()] = func

    return functions


qfunctions = _get_functions()

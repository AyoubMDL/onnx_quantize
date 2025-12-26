import onnx_ir as ir

from onnx_quantize.qfunctions.qgemm import *
from onnx_quantize.qfunctions.qmatmul import *
from onnx_quantize.qfunctions.register import _QFUNCTIONS, OP_TYPES_TO_QUANTIZE


def get_qfunctions():
    """Get all registered quantization functions.

    This function is called dynamically to pick up any functions
    registered after module import (e.g., grouped quantization functions).
    """
    functions = {}
    for func in _QFUNCTIONS:
        func = ir.serde.deserialize_function(func)
        functions[func.identifier()] = func

    return functions

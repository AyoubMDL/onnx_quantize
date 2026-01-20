import onnx_ir as ir
import onnxscript

from onnx_quantize.qfunctions.factory import *
from onnx_quantize.qfunctions.register import (
    _QFUNCTIONS,
    MS_OPSET,
    OP_TYPES_TO_QUANTIZE,
    QUANT_OPSET,
)


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

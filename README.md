# onnx-quantize

**ONNX Quantization Framework** built on top of  
- [ONNX IR](https://github.com/onnx/ir-py)  
- [ONNXScript](https://github.com/microsoft/onnxscript)  

> ⚠️ This project is under active development.

---

## 📦 Installation

Install directly from **PyPI**:

```python
pip install onnx-quantize
```

## 🚀 Quick Start

Here’s a minimal example to quantize an ONNX model:
```python
from onnx_quantize import QConfig, quantize
from onnxruntime.quantization import QuantType
import onnx

# Load your model
model = onnx.load("your_model.onnx")

# Define quantization configuration
qconfig = QConfig(
    is_static=False,
    activations_dtype=QuantType.QInt8,
    activations_symmetric=False,
    weights_dtype=QuantType.QInt8,
    weights_symmetric=True,
    weights_per_channel=False,
)

# Quantize the model
qmodel = quantize(model, qconfig)

# Save the quantized model
onnx.save(qmodel, "qmodel.onnx")
```

🧩 Features (planned)

The goal is to have all of what [Neural compressor](https://github.com/onnx/neural-compressor) have but using 
[ONNXScript](https://github.com/microsoft/onnxscript) and [ONNX IR](https://github.com/onnx/ir-py).
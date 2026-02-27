import onnx
import pytest
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from tqdm import tqdm

from onnx_quantize import AwqConfig, QConfig, QuantType, QWeightArgs, SmoothQuantConfig, quantize
from onnx_quantize.core._qconfig import QActivationArgs


@pytest.mark.parametrize(
    "quant_type, act_dtype, symmetric, is_static, preprocessor, expected_accuracy",
    [
        (QuantType.QUInt8, QuantType.QUInt8, False, False, None, 0.93),
        (QuantType.QUInt8, QuantType.QUInt8, False, True, SmoothQuantConfig(alpha=0.5), 0.93),
        (QuantType.QUInt8, QuantType.QUInt8, False, True, AwqConfig(), 0.93),
        (QuantType.QInt8, QuantType.QInt8, True, True, None, 0.93),
    ],
)
def test_quantize_bert_weights_inputs(
    bert_dataset,
    bert_model,
    bert_onnx_dir,
    bert_calibration_data,
    quant_type,
    act_dtype,
    symmetric,
    is_static,
    preprocessor,
    expected_accuracy,
):
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=quant_type,
            symmetric=symmetric,
            strategy="channel",
        ),
        input_activations=QActivationArgs(
            dtype=act_dtype,
            is_static=is_static,
        ),
        calibration_data=bert_calibration_data,
        preprocessors=[preprocessor] if preprocessor else [],
    )

    # Quantize
    qmodel = quantize(bert_model, qconfig)
    onnx.save(qmodel, bert_onnx_dir / "quantized_model.onnx")

    # Reload with full HF context
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        bert_onnx_dir,
        file_name="quantized_model.onnx",
        provider="CPUExecutionProvider",
    )

    # Evaluate
    correct = 0
    for batch in tqdm(torch.utils.data.DataLoader(bert_dataset, batch_size=8)):
        outputs = onnx_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()

    # float has ~0.94 accuracy
    assert correct / len(bert_dataset) == expected_accuracy

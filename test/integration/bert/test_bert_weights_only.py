import onnx
import pytest
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from tqdm import tqdm

from onnx_quantize import AwqConfig, HqqConfig, QConfig, QuantType, QWeightArgs, quantize


@pytest.mark.parametrize(
    "quant_type, strategy, group_size, algorithm, preprocessors, expected_accuracy",
    [
        (QuantType.QUInt8, "channel", None, None, None, 0.94),
        (QuantType.QUInt4, "group", 128, None, None, 0.93),
        (QuantType.QUInt4, "group", 128, HqqConfig(early_stop=False), None, 0.94),
        (QuantType.QInt8, "channel", None, None, AwqConfig(), 0.94),
    ],
)
def test_quantize_bert_weights_only(
    bert_dataset,
    bert_model,
    bert_onnx_dir,
    bert_calibration_data,
    quant_type,
    strategy,
    group_size,
    algorithm,
    preprocessors,
    expected_accuracy,
):
    qconfig = QConfig(
        weights=QWeightArgs(
            dtype=quant_type,
            symmetric=False,
            strategy=strategy,
            group_size=group_size,
            algorithm=algorithm,
        ),
        preprocessors=[preprocessors] if preprocessors else [],
        calibration_data=bert_calibration_data,
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

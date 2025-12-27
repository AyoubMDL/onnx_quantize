import onnx
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from tqdm import tqdm
from transformers import AutoTokenizer

from onnx_quantize import QConfig, QuantType, quantize


def test_quantize_bert_weights_only(tmp_path):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_name, export=True, provider="CPUExecutionProvider"
    )
    onnx_model.save_pretrained(tmp_path)
    model = onnx.load(tmp_path / "model.onnx")

    qconfig = QConfig(
        weights_only=True, weights_dtype=QuantType.QUInt8, weights_symmetric=True, strategy="tensor"
    )
    qmodel = quantize(model, qconfig)
    onnx.save(qmodel, tmp_path / "quantized_model.onnx")

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        tmp_path,
        file_name="quantized_model.onnx",
        provider="CPUExecutionProvider",
    )

    dataset = load_dataset("glue", "sst2", split="validation[:100]")

    def preprocess(examples):
        return tokenizer(examples["sentence"], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Evaluate
    correct = 0
    for batch in tqdm(torch.utils.data.DataLoader(dataset, batch_size=8)):
        outputs = onnx_model(**{k: v for k, v in batch.items() if k != "label"})
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()

    # TODO: float has about 0.94
    assert correct / len(dataset) == 0.86

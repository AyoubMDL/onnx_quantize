import onnx
import pytest
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@pytest.fixture(scope="module")
def bert_dataset():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    dataset = load_dataset("glue", "sst2", split="validation[:100]")

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding=True,
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


@pytest.fixture(scope="module")
def bert_onnx_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("bert_onnx")

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        _MODEL_NAME,
        export=True,
        provider="CPUExecutionProvider",
    )
    onnx_model.save_pretrained(path)

    return path


@pytest.fixture(scope="module")
def bert_model(bert_onnx_dir):
    """Raw ONNX model for quantization."""
    return onnx.load(bert_onnx_dir / "model.onnx")


@pytest.fixture(scope="module")
def bert_calibration_data(bert_dataset):
    """Calibration data for the BERT model (all required inputs)."""
    return {
        "input_ids": torch.stack([bert_dataset[i]["input_ids"] for i in range(32)]).numpy(),
        "attention_mask": torch.stack(
            [bert_dataset[i]["attention_mask"] for i in range(32)]
        ).numpy(),
    }

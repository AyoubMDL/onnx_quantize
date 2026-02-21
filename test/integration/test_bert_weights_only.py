import onnx
import pytest
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from tqdm import tqdm
from transformers import AutoTokenizer

from onnx_quantize import HqqConfig, QConfig, QuantType, QWeightArgs, quantize


@pytest.fixture(scope="module")
def bert_dataset():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    path = tmp_path_factory.mktemp("bert_onnx")

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider",
    )
    onnx_model.save_pretrained(path)

    return path


@pytest.fixture(scope="module")
def bert_model(bert_onnx_dir):
    """Raw ONNX model for quantization."""
    return onnx.load(bert_onnx_dir / "model.onnx")


@pytest.mark.parametrize(
    "quant_type, strategy, group_size, algorithm, expected_accuracy",
    [
        (QuantType.QUInt8, "channel", None, None, 0.94),
        (QuantType.QUInt4, "group", 128, None, 0.93),
        (QuantType.QUInt4, "group", 128, HqqConfig(early_stop=False), 0.94),
    ],
)
def test_quantize_bert_weights_only(
    bert_dataset,
    bert_model,
    bert_onnx_dir,
    quant_type,
    strategy,
    group_size,
    algorithm,
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

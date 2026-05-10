import numpy as np
import onnx_ir as ir
from datasets import load_dataset
from onnxruntime_genai.models.builder import create_model
from transformers import AutoConfig, AutoTokenizer

from onnx_quantize import AwqConfig, CalibrationParams, QConfig, QWeightArgs, quantize


MODEL_ID = "google/gemma-3-270m"


def make_calibration_data(tokenizer, config, num_samples=32, block_size=512):
    """Build wikitext-2 AWQ calibration feeds for gemma3 (no position_ids input)."""
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    all_ids = tokenizer(text, return_tensors="np").input_ids[0]

    chunks = []
    for i in range(num_samples):
        start = i * block_size
        end = start + block_size
        if end > len(all_ids):
            break
        chunks.append(all_ids[start:end][np.newaxis, :])

    batch = np.concatenate(chunks, axis=0).astype(np.int64)
    n = batch.shape[0]
    data = {
        "input_ids": batch,
        "attention_mask": np.ones((n, block_size), dtype=np.int64),
    }
    empty_kv = np.zeros((n, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for i in range(config.num_hidden_layers):
        data[f"past_key_values.{i}.key"] = empty_kv
        data[f"past_key_values.{i}.value"] = empty_kv
    return data


if __name__ == "__main__":
    output_dir = "gemma3_genai"
    create_model(MODEL_ID, "", output_dir, "fp32", "cpu", ".")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID).get_text_config()

    # Define calibation params for AWQ pass
    num_samples = 64
    block_size = 256
    calibration_data = make_calibration_data(
        tokenizer, config, num_samples=num_samples, block_size=block_size
    )

    # Define quantization config
    qconfig = QConfig(
        weights=QWeightArgs(dtype="uint4", strategy="group", group_size=128),
        preprocessors=[AwqConfig()],
        calibration_data=calibration_data,
        calibration_params=CalibrationParams(batch_size=1, num_samples=num_samples),
        ignore=["lm_head"],
    )

    # Load model
    model = ir.load(f"{output_dir}/model.onnx")

    # Quantize
    qmodel = quantize(model, qconfig)

    # Save model
    ir.save(qmodel, "qgemma.onnx", external_data="qgemma.onnx.data")

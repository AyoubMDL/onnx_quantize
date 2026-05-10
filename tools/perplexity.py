import argparse

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x_max = x.max(axis=-1, keepdims=True)
    x = x - x_max
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _empty_past(config) -> dict[str, np.ndarray]:
    past = {}
    shape = (1, config.num_key_value_heads, 0, config.head_dim)
    for i in range(config.num_hidden_layers):
        past[f"past_key_values.{i}.key"] = np.zeros(shape, dtype=np.float32)
        past[f"past_key_values.{i}.value"] = np.zeros(shape, dtype=np.float32)
    return past


def perplexity_eval(
    model_path: str,
    model_id: str,
    max_length: int = 2048,
    stride: int = 512,
    provider: str = "CPUExecutionProvider",
) -> float:
    """Compute wikitext-2 perplexity for an ONNX causal LM (HF sliding-window method)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id).get_text_config()
    session = ort.InferenceSession(model_path, providers=[provider])
    input_names = {inp.name for inp in session.get_inputs()}

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    input_ids = tokenizer(text, return_tensors="np").input_ids[0]
    seq_len = len(input_ids)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    pbar = tqdm(range(0, seq_len, stride), desc="Perplexity")
    for begin in pbar:
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        chunk = input_ids[begin:end][None, :].astype(np.int64)
        seq = chunk.shape[1]

        feeds = {
            "input_ids": chunk,
            "attention_mask": np.ones((1, seq), dtype=np.int64),
            **_empty_past(config),
        }
        if "position_ids" in input_names:
            feeds["position_ids"] = np.arange(seq, dtype=np.int64)[None, :]

        logits = session.run(["logits"], feeds)[0][0]

        # Shift: logits[t] predicts token[t+1]
        log_probs = _log_softmax(logits[:-1])
        targets = chunk[0, 1:]

        # Only count the newly-revealed tokens of this window
        loss_targets = targets[-trg_len:]
        lp = log_probs[-trg_len:]
        nll = -lp[np.arange(len(loss_targets)), loss_targets]

        total_nll += nll.sum()
        total_tokens += len(nll)
        pbar.set_postfix(ppl=f"{np.exp(total_nll / total_tokens):.4f}")

        prev_end = end
        if end == seq_len:
            break

    ppl = float(np.exp(total_nll / total_tokens))
    print(f"perplexity of {model_path}: {ppl}")
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the ONNX model file.")
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF repo id for tokenizer/config (e.g. google/gemma-3-270m).",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    perplexity_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        max_length=args.max_length,
        stride=args.stride,
        provider=args.provider,
    )

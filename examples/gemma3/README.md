# Gemma 3 (270M)

## Quantize

```bash
# RTN int8
python examples/gemma3/gemma3_rtn.py

# AWQ uint4
python examples/gemma3/gemma3_awq.py
```

## Perplexity

```bash
# fp32 baseline
python tools/perplexity.py \
    --model-path gemma3_genai/model.onnx \
    --model-id google/gemma-3-270m

# quantized
python tools/perplexity.py \
    --model-path qgemma.onnx \
    --model-id google/gemma-3-270m
```

## Results (wikitext-2)

| Model     | Perplexity |
|-----------|------------|
| fp32      | 16.77      |
| RTN int8  | 16.79      |
| AWQ uint4 | 21.24      |

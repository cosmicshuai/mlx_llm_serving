# Gemma 4 E4B — Model Specification

**HuggingFace repo:** `google/gemma-4-E4B`
**Pinned revision:** `7aa32e6889efd6300124851b164f8b364314c3d8`
**License:** Apache-2.0
**Architecture class:** `Gemma4ForConditionalGeneration`
**Scope:** Text-only inference (vision and audio encoders are out of scope for v1)

---

## Repository File Manifest

| File                     | Size     | Purpose                          |
| ------------------------ | -------- | -------------------------------- |
| `config.json`            | 5.1 KB   | Full model config (text + vision + audio) |
| `generation_config.json` | 181 B    | Default sampling parameters      |
| `model.safetensors`      | ~16 GB   | All weights (text + vision + audio) in single file |
| `tokenizer.json`         | ~32 MB   | BPE vocab + merges (HF tokenizers format) |
| `tokenizer_config.json`  | 881 B    | Tokenizer metadata + special tokens |
| `processor_config.json`  | 1.7 KB   | Multimodal processor config (not needed for v1) |

**Single safetensors file** — no sharding. Text-only loading must selectively load text model weight keys and skip vision/audio encoder tensors.

---

## Default Generation Config

| Parameter    | Value  |
| ------------ | ------ |
| `do_sample`  | true   |
| `temperature`| 1.0    |
| `top_k`      | 64     |
| `top_p`      | 0.95   |
| `bos_token_id` | 2   |
| `eos_token_id` | 1   |
| `pad_token_id` | 0   |

These are the model publisher's recommended defaults for sampling.

---

## Text Model Architecture

| Parameter                    | Value                          |
| ---------------------------- | ------------------------------ |
| `model_type`                 | `gemma4_text`                  |
| `num_hidden_layers`          | 42                             |
| `hidden_size`                | 2560                           |
| `intermediate_size`          | 10240                          |
| `num_attention_heads`        | 8                              |
| `num_key_value_heads`        | 2 (GQA, 4:1 ratio)            |
| `head_dim` (sliding)         | 256                            |
| `global_head_dim` (full)     | 512                            |
| `hidden_activation`          | `gelu_pytorch_tanh`            |
| `vocab_size`                 | 262144                         |
| `max_position_embeddings`    | 131072                         |
| `rms_norm_eps`               | 1e-6                           |
| `attention_bias`             | false                          |
| `tie_word_embeddings`        | true                           |
| `dtype`                      | bfloat16                       |

### Per-Layer Embeddings (PLE)

| Parameter                        | Value   |
| -------------------------------- | ------- |
| `hidden_size_per_layer_input`    | 256     |
| `vocab_size_per_layer_input`     | 262144  |

Each layer receives a separate low-dimensional (256-d) per-layer embedding that is concatenated or added to the main hidden state. This gives the model ~4.5B effective parameters from ~8B total.

### Layer Pattern

42 layers with a repeating pattern of 5 sliding-window + 1 global (full) attention:

```
Layers  0– 4: sliding_attention
Layer   5:    full_attention
Layers  6–10: sliding_attention
Layer  11:    full_attention
Layers 12–16: sliding_attention
Layer  17:    full_attention
Layers 18–22: sliding_attention
Layer  23:    full_attention
Layers 24–28: sliding_attention
Layer  29:    full_attention
Layers 30–34: sliding_attention
Layer  35:    full_attention
Layers 36–40: sliding_attention
Layer  41:    full_attention
```

**7 global (full) attention layers:** 5, 11, 17, 23, 29, 35, 41
**35 sliding window attention layers:** all others

### Shared KV Cache

| Parameter              | Value |
| ---------------------- | ----- |
| `num_kv_shared_layers` | 18    |

The last 18 layers (layers 24–41) reuse KV states from earlier layers — they do **not** compute their own K/V projections. They still have Q projections and attention output projections.

This means only layers 0–23 produce fresh KV entries. Layers 24–41 index into the KV cache of their paired earlier layers.

### Dual RoPE Configuration

| Attention Type     | `rope_type`    | `rope_theta` | `partial_rotary_factor` |
| ------------------ | -------------- | ------------ | ----------------------- |
| `sliding_attention`| `default`      | 10,000       | 1.0 (implicit)          |
| `full_attention`   | `proportional` | 1,000,000    | 0.25                    |

- Sliding window layers: standard RoPE with theta=10k, applied to full head dimension
- Global layers: proportional RoPE (p-RoPE) with theta=1M, applied to 25% of head dimension

### Sliding Window

| Parameter        | Value |
| ---------------- | ----- |
| `sliding_window` | 512   |

Sliding attention layers attend only to the most recent 512 tokens.

### Output Logit Softcapping

| Parameter                | Value |
| ------------------------ | ----- |
| `final_logit_softcapping`| 30.0  |

Logits are capped via `logits = cap * tanh(logits / cap)` before the softmax.

### MoE

Not enabled (`enable_moe_block: false`).

---

## Tokenizer

| Property            | Value                                    |
| ------------------- | ---------------------------------------- |
| Tokenizer class     | `GemmaTokenizer`                         |
| Backend             | HuggingFace `tokenizers` (Rust BPE)      |
| Format              | `tokenizer.json` (NOT SentencePiece `.model`) |
| Vocab size          | 262,144                                  |
| Padding side        | left                                     |

### Special Tokens (Text-Relevant)

| Token               | Symbol              | ID     |
| ------------------- | ------------------- | ------ |
| BOS                 | `<bos>`             | 2      |
| EOS                 | `<eos>`             | 1      |
| PAD                 | `<pad>`             | 0      |
| UNK                 | `<unk>`             | —      |
| Start of turn       | `<\|turn>`          | —      |
| End of turn         | `<turn\|>`          | —      |
| Start of tool call  | `<\|tool_call>`     | —      |
| End of tool call    | `<tool_call\|>`     | —      |
| Start of tool resp  | `<\|tool_response>` | —      |
| End of tool resp    | `<tool_response\|>` | —      |
| Think               | `<\|think\|>`       | —      |

### Tokenizer Implementation Note

**SentencePiece (`tokenizer.model`) is NOT shipped with this model.** The tokenizer is defined exclusively in `tokenizer.json` using the HuggingFace `tokenizers` library format (Rust-based BPE).

For C++ integration, options are:
1. **HuggingFace `tokenizers` C++ bindings** — most faithful, loads `tokenizer.json` directly
2. **Custom BPE loader** — parse `tokenizer.json` vocab + merges, implement BPE encode/decode
3. **Convert to SentencePiece** — lossy, not recommended

Recommended: option 1 or 2. The `tokenizers` Rust library has a C FFI that can be linked from C++.

---

## Multimodal Components (Out of Scope for v1)

Documented here for completeness; these are **not** implemented in Phase 1.

### Vision Encoder
- 16 layers, ViT-style, patch_size=16, hidden_size=768
- Produces 280 soft tokens per image (`vision_soft_tokens_per_image`)

### Audio Encoder
- 12 layers, hidden_size=1024, output projected to 1536-d

### Special Token IDs (Multimodal)
- `image_token_id`: 258880
- `audio_token_id`: 258881
- `eoi_token_id`: 258882
- `eoa_token_id`: 258883
- `video_token_id`: 258884

---

## Implications for Plan

1. **Plan `plan.md:119` assumes SentencePiece — this is incorrect.** The tokenizer interface must support loading from `tokenizer.json` (BPE format). SentencePiece is not an option for Gemma 4 E4B.

2. **Shared KV layers reduce KV cache memory by ~43%.** Only 24 of 42 layers need KV storage. The paged KV cache manager (Phase 2) must account for this — block allocation should be per-layer, not uniform across all 42 layers.

3. **Dual head dimensions complicate attention.** Sliding layers use head_dim=256, global layers use head_dim=512. The attention kernel and KV cache block size must handle both.

4. **PLE adds a second embedding table.** Model loading must handle both the main embedding (262144 x 2560) and per-layer embeddings (262144 x 256 per layer, or a shared variant).

5. **Logit softcapping** must be applied after the final linear projection, before sampling.

6. **GQA ratio is 4:1** (8 Q heads, 2 KV heads). The attention implementation must broadcast KV heads.

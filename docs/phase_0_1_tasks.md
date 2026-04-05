# Phase 0+1 Task List

**Scope:** From empty repo to Gemma 4 E4B generating tokens via CLI.
**Model spec:** [docs/models/gemma4_e4b.md](./models/gemma4_e4b.md)

---

## Task 1 — Project Scaffold & CMake

**Goal:** Repository builds a hello-world binary on Apple Silicon with all dependencies resolved.

- Initialize git repo
- Create directory structure:
  ```
  src/
    core/         # types, interfaces, configs
    model/        # model graph, layers, forward pass
    tokenizer/    # tokenizer wrapper
    loader/       # safetensors parser, weight mapping
    sampling/     # sampling strategies
    serving/      # (empty, Phase 9)
    scheduler/    # (empty, Phase 6)
  include/        # public headers
  tests/          # GoogleTest suites
  third_party/    # git submodules
  docs/
    architecture/
    models/
  benchmarks/
  ```
- `CMakeLists.txt` at root — fetch/link all dependencies:
  - MLX (C++ API)
  - mlc-ai/tokenizers-cpp (submodule, brings Rust build)
  - google/minja (header-only)
  - nlohmann/json (header-only)
  - spdlog
  - CLI11 (header-only)
  - GoogleTest
- Verify: `cmake --build` produces a binary that prints "hello" and exits 0
- Verify: Rust toolchain compiles tokenizers-cpp into static lib
- Verify: MLX links and can create a trivial `mlx::core::array`

**Done when:** `./llm-serve` binary compiles, links all deps, runs on Apple Silicon.

---

## Task 2 — Architecture Diagram

**Goal:** Visual map of all components and data flow committed to `docs/architecture/`.

- Mermaid `flowchart TD` showing:
  - CLI → Engine → Model → MLX ops
  - Config parser, weight loader, tokenizer as inputs
  - KV cache as internal state
  - Sampling as output stage
- Highlight Phase 0+1 scope vs future phases
- Commit as `docs/architecture/system_overview.md`

**Done when:** Diagram reviewed, committed, matches the component structure in Task 1.

---

## Task 3 — Core Types & Interfaces

**Goal:** All inter-component contracts defined. Any component can be implemented independently.

- `TextModelConfig` — parsed from `config.json`, covers all fields from [gemma4_e4b.md](./models/gemma4_e4b.md): layer_types, head dims, RoPE params, shared KV count, PLE dims, softcapping, etc.
- `Tokenizer` interface — `encode(string) → vector<int>`, `decode(vector<int>) → string`, `vocab_size()`, `eos_id()`, `bos_id()`
- `ChatTemplateRenderer` interface — `render(messages, tools) → string`
- `CheckpointLoader` interface — `load(path, config) → ModelWeights`
- `SamplingConfig` — temperature, top_k, top_p, max_tokens, stop_token_ids
- `GenerationResult` — token_ids, text, finish_reason, timing stats
- `ModelRunner` interface — `prefill(tokens, kv_cache) → logits`, `decode_step(token, kv_cache) → logits`

**Done when:** All headers compile, doc comments on every public method, no Gemma-specific types leak through interfaces.

---

## Task 4 — Config Parser

**Goal:** Parse Gemma 4 E4B's `config.json` into `TextModelConfig` struct.

- Load JSON via nlohmann/json
- Extract `text_config` section (skip vision/audio)
- Parse `layer_types` array into enum vector
- Parse `rope_parameters` per attention type
- Parse PLE, shared KV, GQA, softcapping fields
- Unit tests: parse the pinned config, assert all fields match expected values
- Error handling: missing fields, wrong types → clear error messages

**Done when:** `TextModelConfig` round-trips through parse with all fields from gemma4_e4b.md spec.

---

## Task 5 — Tokenizer Integration

**Goal:** Encode and decode text using Gemma 4's tokenizer, validated against reference output.

- Wrap `mlc-ai/tokenizers-cpp` behind our `Tokenizer` interface
- Load `tokenizer.json` via `Tokenizer::FromBlobJSON()`
- Implement `encode()`, `decode()`, `vocab_size()`, `eos_id()`, `bos_id()`
- Handle special tokens: BOS prepend, EOS detection
- Unit tests:
  - Encode known strings, compare token IDs against HF `transformers` reference output
  - Decode token IDs back to original strings
  - Vocab size == 262144
  - EOS == 1, BOS == 2, PAD == 0
- Edge cases: empty string, Unicode, very long input

**Done when:** Encode/decode matches HF `transformers` tokenizer output for a test suite of 20+ strings.

---

## Task 6 — Chat Template

**Goal:** Render Gemma 4 chat messages into the correct prompt format.

- Integrate `google/minja` (header-only, add to include path)
- Load Gemma 4's chat template (from `tokenizer_config.json` or hardcoded for Phase 1)
- Implement `ChatTemplateRenderer` for Gemma 4
- Support: system message, user/assistant turns, multi-turn conversations
- Golden tests: known multi-turn conversations → expected prompt string
- Raw `--prompt` mode bypasses chat template entirely

**Done when:** Multi-turn chat renders match HF `transformers` `apply_chat_template()` output.

---

## Task 7 — Safetensors Loader & Weight Mapping

**Goal:** Load Gemma 4 E4B weights from `model.safetensors`, mapped to our model parameter names.

- Implement safetensors format parser:
  - Read header (JSON metadata at file start)
  - Memory-map tensor data (no full copy into RAM)
  - Support bf16 dtype
- Gemma 4 weight name mapping table:
  - HF name → internal name (e.g., `model.layers.0.self_attn.q_proj.weight` → our param path)
  - Filter: skip all `vision_tower.*` and `audio_tower.*` keys
  - Validate: no unmapped text model keys left over
- Fuse Q/K/V projections at load time if beneficial for MLX matmul
- Unit tests:
  - Parse header from a real safetensors file, verify tensor names and shapes
  - Weight mapping covers all text model keys
  - Skipped keys are exactly the vision/audio keys

**Done when:** All text model tensors load into `mlx::core::array` with correct shapes and dtype.

---

## Task 8 — Model Building Blocks (MLX Ops)

**Goal:** Individual layers produce correct output when tested in isolation.

- **RMSNorm**: `x * rsqrt(mean(x², dim=-1) + eps) * weight`, eps=1e-6
- **RoPE**:
  - Standard RoPE (theta=10k, full head dim) for sliding layers
  - Proportional RoPE (theta=1M, partial_rotary_factor=0.25) for global layers
  - Must handle both head_dim=256 and head_dim=512
- **Activation**: `gelu_pytorch_tanh` (GELU with tanh approximation)
- **Embedding**: main embedding (262144 × 2560) + PLE embedding (262144 × 256)
- All implemented as functions over `mlx::core::array`
- Unit tests for each:
  - RMSNorm: compare against PyTorch reference output
  - RoPE: compare against HF transformers RoPE for both variants
  - Activation: compare against `torch.nn.functional.gelu(x, approximate='tanh')`

**Done when:** Each op matches reference output within bf16 tolerance (atol=1e-2, rtol=1e-2).

---

## Task 9 — Attention Layer

**Goal:** Single attention layer produces correct output for both sliding window and global attention.

- GQA attention with 8 Q heads, 2 KV heads (broadcast KV)
- Sliding window attention: head_dim=256, standard RoPE, attend to last 512 tokens only
- Global (full) attention: head_dim=512, proportional RoPE, attend to all tokens
- Causal masking
- Naive KV cache integration: append K/V to pre-allocated contiguous buffer per layer
- Shared KV: layers 24–41 read from earlier layer's cache slot (no own K/V projection)
- Unit tests:
  - Single-layer forward with known input → compare against HF transformers layer output
  - Sliding window masking: tokens beyond window get -inf
  - Shared KV layer correctly reads from source layer's cache

**Done when:** Both attention variants produce correct output, shared KV layers reference the right cache.

---

## Task 10 — MLP & Transformer Block

**Goal:** Full transformer block (attention + MLP + norms + PLE) produces correct output.

- Gated MLP: `down_proj(act(gate_proj(x)) * up_proj(x))` with gelu_pytorch_tanh
  - gate_proj: 2560 → 10240
  - up_proj: 2560 → 10240
  - down_proj: 10240 → 2560
- Transformer block: pre-norm → attention → residual → pre-norm → MLP → residual
- PLE: per-layer embedding lookup → project/add to hidden state at layer input
- Compose into a `TransformerBlock` that takes config to determine attention type
- Unit tests:
  - Full block forward matches HF transformers single-layer output
  - PLE contribution is non-zero and correctly shaped

**Done when:** Single transformer block matches reference at bf16 tolerance.

---

## Task 11 — Full Model Assembly & Generation Loop

**Goal:** Stack all 42 layers, generate tokens autoregressively from a prompt.

- Model assembly:
  - Token embedding (262144 × 2560, tied with LM head)
  - 42 × TransformerBlock, configured per `layer_types[]`
  - Final RMSNorm
  - LM head (tied weights → transpose of embedding)
  - Logit softcapping: `30.0 * tanh(logits / 30.0)`
- Load weights from Task 7 into assembled model
- Prefill: process full prompt in one forward pass, populate KV cache
- Decode loop: one token at a time, append to KV cache, sample, repeat until EOS or max_tokens
- Sampling: greedy (argmax) and temperature-based (softmax → multinomial)
- Test: load real Gemma 4 E4B, run a prompt, inspect output for coherence

**Done when:** Model generates coherent multi-sentence text from a prompt.

---

## Task 12 — Correctness Verification

**Goal:** Greedy decoding output matches reference implementation token-for-token.

- Create a verification prompt suite (5–10 prompts of varying length)
- Run same prompts through HF `transformers` with greedy decoding, record token-by-token output
- Run through our implementation with greedy decoding
- Compare: must match token-for-token (greedy is deterministic)
- Verify specifically:
  - Sliding window layers: attention pattern is windowed
  - Global layers: attention pattern is full
  - Shared KV layers: read from correct source layer
  - Dual RoPE: correct theta and partial factor per layer type
  - PLE: per-layer embeddings contribute to hidden state
  - Logit softcapping: applied before argmax

**Done when:** 100% token match on all verification prompts vs HF transformers reference.

---

## Task 13 — CLI Interface

**Goal:** User-facing CLI for interactive inference.

- CLI11 argument parser:
  - `--model <path>` (required): path to HuggingFace model directory
  - `--prompt <text>`: raw prompt mode (bypasses chat template)
  - `--chat`: interactive multi-turn chat mode (uses chat template)
  - `--temperature <float>`: sampling temperature (default: 1.0)
  - `--top-k <int>`: top-k sampling (default: 64)
  - `--top-p <float>`: nucleus sampling (default: 0.95)
  - `--max-tokens <int>`: max tokens to generate (default: 512)
  - `--seed <int>`: random seed for reproducibility
- Print generated tokens to stdout as they're produced (streaming feel)
- Print timing stats after generation: TTFT, tokens/sec, total tokens, peak memory
- Structured logging via spdlog (to stderr, not mixed with output)

**Done when:** `./llm-serve --model ./gemma-4-E4B --prompt "Hello"` produces streamed output with timing stats.

---

## Task 14 — Benchmark Harness & Baseline

**Goal:** Reproducible benchmark suite with recorded baseline numbers.

- Benchmark harness:
  - Fixed prompt suite (short: 32 tokens, medium: 256, long: 1024)
  - Measure: TTFT, decode tok/s, peak memory (via `mlx::core::metal::get_peak_memory()`)
  - Report: table to stdout, JSON to file
  - Multiple runs with statistical summary (median, p95)
- Record environment in `benchmarks/reference_env.md`:
  - Mac model, chip, unified memory
  - macOS version, Xcode version
  - MLX revision, model revision (pinned commit SHA)
  - Prompt suite checksums
- Record baseline in `benchmarks/baseline.md`:
  - All metrics, absolute values
  - Comparison vs RFC NFR1 targets

**Done when:** `benchmarks/baseline.md` and `benchmarks/reference_env.md` committed with real numbers from the reference machine.

---

## Dependency Graph

```
T1 (scaffold/CMake)
├── T2 (architecture diagram)
├── T3 (types & interfaces)
│   ├── T4 (config parser)
│   ├── T5 (tokenizer)
│   ├── T6 (chat template)
│   ├── T7 (safetensors loader)
│   └── T8 (model building blocks)
│       └── T9 (attention)
│           └── T10 (MLP + transformer block)
│               └── T11 (full model + generation)
│                   ├── T12 (correctness verification)
│                   ├── T13 (CLI)
│                   └── T14 (benchmark)
```

T4, T5, T6, T7, T8 can proceed in parallel after T3.
T11 is the integration point — depends on T4 + T5 + T7 + T10.
T12, T13, T14 can proceed in parallel after T11.

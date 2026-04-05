# Implementation Plan: LLM Serving on Apple Silicon

**RFC:** [llm_serving_mac_rfc.md](./llm_serving_mac_rfc.md)
**Created:** 2026-04-04
**Last Updated:** 2026-04-04

---

## Approach

1. Design components and data contracts upfront
2. Build a minimal E2E flow (MVP) with placeholder optimizations
3. Validate MVP correctness and establish baseline benchmarks
4. Replace placeholders with real optimizations one at a time, measuring impact

## Cross-Cutting Concerns

### Third-Party Dependencies

| Concern          | Library                  | Rationale                                                   |
| ---------------- | ------------------------ | ----------------------------------------------------------- |
| Compute backend  | MLX (C++ API)            | Native Apple Silicon, Metal acceleration                    |
| Tokenizer        | mlc-ai/tokenizers-cpp    | Wraps HF `tokenizers` Rust lib via C FFI; loads `tokenizer.json` directly; CMake subdir integration; also supports SentencePiece `.model` files |
| Chat templates   | google/minja             | Header-only C++ Jinja subset for LLM chat templates; used by llama.cpp, GPT4All; depends only on nlohmann/json |
| HTTP server      | cpp-httplib              | Header-only, lightweight, widely used in llama.cpp/vLLM     |
| JSON             | nlohmann/json            | Industry standard C++ JSON library                          |
| Testing          | GoogleTest               | Standard C++ test framework                                 |
| Metrics          | prometheus-cpp           | Prometheus client library for C++                           |
| Logging          | spdlog                   | Fast structured logging, widely adopted                     |
| CLI parsing      | CLI11                    | Header-only, modern C++ argument parser                     |
| Build system     | CMake                    | MLX uses CMake, natural fit                                 |

### Tokenizer, Chat Template & Model Format Strategy

- **Abstractions first:** Define `Tokenizer`, `ChatTemplateRenderer`, `ModelConfigLoader`, and `CheckpointLoader` interfaces in Phase 0. Gemma-only code in Phase 1 should be the first concrete implementation, not a hard-wired assumption baked into the whole system.
- **Tokenizer support:** Phase 1 uses `mlc-ai/tokenizers-cpp` to load Gemma 4's `tokenizer.json` (HF BPE format). This library wraps the official HF `tokenizers` Rust implementation via C FFI, ensuring encoding fidelity. It also supports SentencePiece `.model` files for models that ship them (e.g., LLaMA). Requires Rust toolchain at build time (compiles to static `.a` lib). The `Tokenizer` interface defined in Phase 0 abstracts over this — a custom BPE loader can replace `tokenizers-cpp` later without changing downstream code.
- **Chat templates:** Use `google/minja` — a header-only C++ Jinja subset purpose-built for LLM chat templates, battle-tested in llama.cpp and GPT4All. Each newly supported architecture must ship with golden template tests, and raw `--prompt` mode remains the escape hatch.
- **Model formats:** Treat HuggingFace `safetensors` as the source-of-truth load path. Treat GGUF, AWQ, and GPTQ as ingestion formats that are converted into the server's internal representation at import/load time instead of carrying a separate runtime execution path for every external layout.
- **Hot-reload scope:** v1 hot-reload means atomically replacing the currently served model inside one process without restarting it. Concurrent multi-model residency remains a Phase 10 extension.

### Benchmark Regression Rule

Every phase from Phase 2 onward MUST:
1. Run the Phase 1 baseline benchmark suite (same model, same prompts, same hardware)
2. Record results in `benchmarks/phase_N.md`
3. Include a comparison table vs Phase 1 baseline and vs previous phase
4. Flag any regression > 5% in TTFT or decode throughput with a root cause explanation

### Reference Benchmark Matrix

- Before recording the Phase 1 baseline, pin the benchmark environment in `benchmarks/reference_env.md`: exact Mac SKU, unified memory size, macOS version, Xcode/Metal toolchain, MLX revision, model revision, tokenizer assets, prompt suite, and concurrency mix.
- Define one primary release-gate machine and at least one optional secondary regression machine.
- Every benchmark artifact must report both absolute values and deltas versus the RFC targets, not just relative improvements versus the previous phase.

### Multi-Architecture Support

Model architecture support expands incrementally, not as a dedicated phase:
- **Phase 1:** Gemma 4 family only (text-only mode)
- **Phase 4 (Quantization):** Add LLaMA (validates the architecture abstraction)
- **Phase 6 (Continuous Batching):** Add Mistral and Phi (validates that batching works across architectures)
- **Phase 9 (Serving):** All 5 target architectures (Gemma, LLaMA, Mistral, Phi, Qwen) supported

Each new architecture addition requires:
- [ ] Greedy decoding output matches reference implementation
- [ ] Benchmark numbers recorded alongside existing architectures

### Sampling & Request Controls Rollout

- **Phase 1:** greedy decoding, temperature, EOS handling, and max token limits in the CLI path.
- **Phase 5:** add top-k, top-p, stop sequences, and request-scoped decode configuration to the core engine.
- **Phase 6:** integrate request priorities, preemption, and token budgeting with the scheduler.
- **Phase 9:** expose temperature, top-k, top-p, repetition penalty, stop sequences, timeout, and cancellation through the HTTP API.

### RFC Coverage & v1 Exit Gates

| Requirement | Planned Phase(s) | Exit Artifact |
| ----------- | ---------------- | ------------- |
| FR1 — Model Loading | 1, 4, 9, 10 | Loader tests, quantized import tests, hot-reload test |
| FR2 — Inference API | 5, 9 | Decode-config tests, HTTP contract tests |
| FR3 — Inference Optimizations | 2, 3, 6, 7, 8 | Phase benchmarks and correctness suites |
| FR4 — Request Management | 5, 6, 9 | Scheduler tests, timeout/cancellation tests |
| NFR1 — Performance | 1, 6, 9 | Baseline artifact and v1 release-gate report |
| NFR2 — Platform | 0, 1 | Apple Silicon/macOS build and run validation |
| NFR3 — Reliability | 6, 9 | Memory-pressure, hot-reload, timeout, and cancellation tests |
| NFR4 — Observability | 9 | Metrics, tracing, and structured-log validation |
| NFR5 — Extensibility | 0, 4, 6, 10 | Interface reviews and plug-in architecture checks |

- Phase 9 is the v1 sign-off point for the RFC goals.
- v1 is not complete until the primary benchmark machine demonstrates TTFT < 200ms for 1K prompt tokens, > 30 tokens/sec single-request decode, > 200 tokens/sec aggregate batched decode at 8+ concurrent requests, and > 90% usable KV-cache memory utilization, or the RFC is explicitly amended.

---

## Phase 0 — Component Design & Data Contracts

### Desired State
- Architecture diagram (Mermaid) showing all components, their boundaries, and data flow
- Data contracts defined for every inter-component boundary (C++ interfaces / abstract classes)
- Tokenizer, chat template, model config, checkpoint loader, sampling config, scheduler, and metrics sink interfaces defined with clear ownership boundaries
- Stub implementations for all components (compile and link, return mock/hardcoded data)
- Directory structure and CMake build system established

### Boundaries
- NO business logic — only types, interfaces, and stubs
- NO model loading or tensor operations
- NO HTTP server

### Verification
- [ ] Project builds with CMake on macOS (Apple Silicon)
- [ ] All components link together and produce a runnable binary (prints "hello" and exits)
- [ ] Every public interface has a doc comment describing responsibility, inputs, and outputs
- [ ] `SamplingConfig`, `RequestContext`, `ModelDescriptor`, and tokenizer/template interfaces are defined without leaking Gemma-specific types across core boundaries
- [ ] Architecture diagram committed to `docs/architecture/`

---

## Phase 1 — MVP (Naive E2E Inference)

### Desired State
- Load Gemma 4 E4B (text-only mode) from HuggingFace safetensors
- Load tokenizer via `mlc-ai/tokenizers-cpp` from `tokenizer.json` (262K vocab, HF BPE format)
- Implement Gemma 4 text architecture:
  - 42-layer decoder-only transformer
  - Hybrid attention: interleaved sliding window (512 tokens) + global attention
  - Proportional RoPE (p-RoPE) positional encoding
  - Per-Layer Embeddings (PLE): 4.5B effective / 8B total parameters
  - Unified Keys and Values in global layers
  - Shared KV Cache: last N layers reuse K/V states from earlier layers (no own K/V projections)
  - Dual RoPE: standard RoPE for sliding window layers, proportional RoPE for global layers
- Tokenize input prompt, run prefill, decode tokens autoregressively until EOS or max length
- Naive KV cache: pre-allocated, fixed-size, single sequence
- No batching — one request at a time, synchronous
- Basic sampling: greedy and temperature-based, plus EOS and max-token stopping
- Benchmark harness: measures TTFT, decode tok/s, peak memory, and reports to stdout
- CLI interface: `./llm-serve --model <path> --prompt "..."`

### Boundaries
- Single model architecture only (Gemma 4, text-only)
- No image/audio/video encoders — skip multimodal components entirely
- No quantization — bf16 weights only
- No concurrent requests
- No HTTP server — CLI only
- KV cache is monolithic (one contiguous allocation per sequence)

### Verification
- [ ] Model loads successfully from HuggingFace safetensors (text model weights only)
- [ ] Generates coherent text for basic prompts (manual inspection)
- [ ] Output matches reference implementation (MLX or HuggingFace transformers) token-for-token with greedy decoding and same seed
- [ ] Both sliding window and global attention layers produce correct output
- [ ] Shared KV layers correctly read K/V from earlier layers (not computing their own)
- [ ] Dual RoPE: sliding layers use standard RoPE, global layers use proportional RoPE
- [ ] Benchmark harness prints: TTFT, tokens/sec, peak memory usage
- [ ] Baseline numbers recorded in `benchmarks/baseline.md`
- [ ] Benchmark environment recorded in `benchmarks/reference_env.md`

---

## Phase 2 — KV Cache (Paged Memory Manager)

### Desired State
- Block-based KV cache: memory divided into fixed-size blocks (e.g., 16 tokens per block)
- Block allocator: alloc, free, ref-count for future sharing
- Sequence-to-block mapping: each sequence holds an ordered list of block IDs
- KV cache manager that handles grow (allocate new block when current is full) and typed exhaustion handling; admission control and preemption policy remain Phase 6 concerns
- Swap to CPU memory when GPU blocks exhausted (optional, stretch goal)

### Boundaries
- Attention kernel still uses naive implementation (gathers blocks into contiguous tensor before attention)
- Single sequence at a time (no batching yet)
- No prefix sharing yet

### Verification
- [ ] Block allocator unit tests: alloc, free, double-free detection, exhaustion handling
- [ ] Sequence block mapping tests: grow, shrink, random access by token position
- [ ] E2E: same model, same prompt produces identical output as Phase 1 (correctness regression)
- [ ] Memory usage is within 1 block size of optimal (no significant waste)
- [ ] Benchmark comparison vs Phase 1 baseline (expect roughly equivalent perf, no regression > 5%)

---

## Phase 3 — Paged Attention (Metal Kernel)

### Desired State
- Custom Metal kernel for paged attention: reads KV from non-contiguous blocks directly
- Eliminates the gather step from Phase 2 (no copy into contiguous buffer)
- Supports variable sequence lengths within the block table
- Fused softmax + attention in a single kernel pass is an optimization target inside the phase, not a prerequisite for functional completion

### Boundaries
- Single sequence only (no batched attention across sequences yet)
- No FlashAttention-level tiling optimizations (functional correctness first, optimize later)
- fp16 compute only

### Verification
- [ ] Kernel unit tests: compare output against naive attention (torch/MLX reference) within defined fp16/bf16 tolerances
- [ ] Test with varying sequence lengths: short (32), medium (512), long (4096)
- [ ] E2E correctness: pinned deterministic prompt suite matches Phase 2 at the token level; tensor-level comparisons use tolerance thresholds
- [ ] Benchmark: expect improvement in decode latency for long sequences (no gather copy)
- [ ] Memory benchmark: peak memory should be lower (no contiguous copy buffer)

---

## Phase 4 — Quantization

### Desired State
- W4A16 quantization: 4-bit weights, 16-bit activations
- W8A8 quantization: 8-bit weights, 8-bit activations
- Load pre-quantized models (GGUF, AWQ, GPTQ) from HuggingFace by converting them into a canonical internal quantized representation
- On-the-fly quantization: load fp16 model and quantize at startup
- Quantized matmul kernels (Metal) for each scheme
- Dequantization fused into the compute kernel (no separate dequant pass)

### Boundaries
- Quantization applies to linear layers only (not embeddings, not LayerNorm)
- No mixed-precision strategies (uniform quantization across all linear layers)
- No quantization-aware training or calibration (post-training quantization only)
- External format support does not require maintaining a distinct execution backend per on-disk format

### Verification
- [ ] Quantized model output quality: perplexity within acceptable delta vs fp16 on a reference dataset (e.g., WikiText-2)
- [ ] W4A16 7B model fits in < 4GB memory
- [ ] Benchmark: W4A16 decode throughput > 1.5x fp16 (less memory bandwidth)
- [ ] Import representative GGUF, AWQ, and GPTQ checkpoints into the canonical internal layout and produce coherent output
- [ ] Round-trip test: quantize fp16 model → save → load → inference matches direct quantized inference

---

## Phase 5 — Batched Prefill

### Desired State
- Batch multiple prefill requests together in a single forward pass
- Pad or pack sequences to handle variable prompt lengths efficiently
- Paged attention kernel extended to process multiple sequences in one kernel dispatch
- Simple static batching: collect N requests, prefill together, then decode sequentially
- Request-scoped decode configuration carried with each sequence: top-k, top-p, stop sequences, and max token budget

### Boundaries
- No iteration-level scheduling yet — batch is formed once and processed as a unit
- No dynamic batch membership changes mid-decode
- No preemption

### Verification
- [ ] Batched prefill output matches sequential prefill for same inputs (correctness)
- [ ] Prefill throughput: batching 4 prompts is > 2x faster than 4 sequential prefills
- [ ] Paged attention kernel handles multiple block tables in a single dispatch
- [ ] Memory usage scales linearly with batch size (no hidden quadratic allocations)
- [ ] Top-k/top-p configuration is honored per request and stop sequences terminate generation at the correct boundary
- [ ] Benchmark comparison vs Phase 4 recorded in `benchmarks/phase_5.md`

---

## Phase 6 — Continuous Batching

### Desired State
- Iteration-level scheduler: at each decode step, decide which sequences to include in the batch
- Request queue with explicit priority levels: incoming requests enqueued, scheduler pulls them into the active batch by priority first and FCFS within each priority
- Dynamic batch composition: new requests join mid-decode, completed requests leave
- Preemption policy: when memory is full, preempt the lowest-priority active request; newest request within that priority is chosen first to minimize wasted work
- Priority aging prevents starvation for long-waiting low-priority requests
- Builds on Phase 5's batched attention kernel — extends it with dynamic batch membership

### Boundaries
- No HTTP server yet — requests submitted via programmatic API or CLI
- No prefix caching (each request builds KV from scratch)
- Preemption is recompute-based in the v1 path; CPU swap remains optional
- Scheduler APIs must support pluggable decode strategies so speculative decoding can land in Phase 7 without creating a separate request path

### Verification
- [ ] Scheduler unit tests: enqueue, dequeue, preempt, resume, priority ordering, and aging behavior
- [ ] Correctness: 8 concurrent requests produce identical output to running them sequentially
- [ ] Throughput: batched aggregate tokens/sec > 3x single-request throughput (7B model, 8 concurrent) and tracked against the RFC target of > 200 tokens/sec aggregate on the reference machine
- [ ] Preemption test: submit requests exceeding memory capacity, verify graceful eviction and eventual completion
- [ ] No request starvation: all submitted requests complete within bounded time

---

## Phase 7 — Speculative Decoding

### Desired State
- Draft model support: load a smaller model (e.g., 0.5B) alongside the target model (7B)
- Draft-then-verify loop: draft model generates k candidate tokens, target model verifies in one forward pass
- Token tree management: handle branching when partial acceptance occurs
- Acceptance sampling: correct the distribution to match target model exactly
- Configurable speculation length (k)
- Speculative decoding is implemented as a decode strategy inside the Phase 6 scheduler so speculative and standard requests share admission, cancellation, and observability paths
- Scheduler may fall back to standard decoding when batch composition or model pairing makes speculative execution inefficient or invalid

### Boundaries
- Draft and target model must share the same tokenizer
- Initial production scope: only requests sharing the same target/draft pair may be co-batched speculatively; incompatible requests fall back to standard decoding inside the same scheduler
- No self-speculative decoding (separate draft model required)

### Verification
- [ ] Output distribution matches target-only decoding (statistical test over 1000+ samples)
- [ ] Acceptance rate tracking: log and report average acceptance rate
- [ ] Benchmark: effective tokens/sec > 1.5x non-speculative decoding (with good draft model)
- [ ] Correctness: greedy speculative decoding produces identical output to greedy standard decoding
- [ ] Scheduler fallback path: mixed speculative and non-speculative workloads complete correctly without separate request plumbing
- [ ] Fallback: if draft model is unavailable, gracefully degrade to standard decoding

---

## Phase 8 — Prefix Caching

### Desired State
- Hash-based block identification: hash the token content of each KV block
- Block sharing: if a new request's prefix matches cached blocks, reuse them (increment ref count)
- Eviction policy: LRU eviction of unreferenced prefix blocks when memory is tight
- Common prefix detection across requests (e.g., shared system prompt)

### Boundaries
- Read-only sharing (shared blocks are immutable, copy-on-write if mutation needed)
- No cross-model sharing
- No persistent cache across server restarts

### Verification
- [ ] Unit tests: hash computation, block lookup, ref counting, eviction
- [ ] 10 requests with identical system prompt: KV blocks for system prompt allocated only once
- [ ] TTFT improvement: repeated prefix TTFT < 20% of first-request TTFT
- [ ] Correctness: shared-prefix output identical to independent-prefix output
- [ ] Memory test: prefix sharing reduces peak memory proportional to shared prefix length

---

## Phase 9 — Serving Layer & Observability

### Desired State
- HTTP server with OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`)
- SSE streaming for token-by-token responses
- Configurable sampling via API: temperature, top-k, top-p, repetition penalty, stop sequences
- Single-model hot-reload without process restart: load, validate, warm, and atomically swap the active model
- Request timeout, cancellation, and token budget enforcement
- `/health` and `/ready` endpoints
- Prometheus-compatible `/metrics` endpoint
- Structured JSON logging with request correlation IDs
- Per-request tracing: TTFT, decode latency, tokens generated, cache hit rate

### Boundaries
- No authentication or rate limiting (delegate to reverse proxy)
- No TLS termination (delegate to reverse proxy)
- No WebSocket support (SSE only)
- Single-server only (no load balancing coordination)
- Hot-reload replaces one active model at a time; concurrent residency of multiple active models remains Phase 10

### Verification
- [ ] OpenAI-compatible HTTP contract tests pass for `/v1/chat/completions` and `/v1/completions`
- [ ] Smoke test with a pinned `openai-python` client version using the current `client.chat.completions.create(...)` API
- [ ] Streaming: tokens arrive incrementally (not buffered until completion)
- [ ] Sampling controls map correctly from HTTP request to decode behavior, including repetition penalty and stop sequences
- [ ] Hot-reload swaps to a new model without process restart; in-flight requests complete on the old model and new requests use the new model after readiness flips
- [ ] Concurrent load test: 20 concurrent requests, no crashes, all complete correctly
- [ ] Timeout: request exceeding timeout returns appropriate error, resources freed
- [ ] Cancellation: client disconnect frees KV cache blocks within 1 second
- [ ] Metrics endpoint returns valid Prometheus format with all specified metrics
- [ ] Structured logs contain correlation ID traceable across the request lifecycle
- [ ] Release-gate report demonstrates the RFC NFR1 targets on the primary benchmark machine, or records an explicit gap and RFC amendment decision

---

## Phase 10 — Multi-Model Serving (Extension)

### Desired State
- Model registry: load/unload multiple models at runtime via API (`/v1/models`)
- Per-request model routing: `"model"` field in API request selects which model to use
- Shared memory pool: unified block allocator across all loaded models with per-model memory budgets and limits
- Model lifecycle management: load, unload, swap models without server restart
- Dynamic LoRA adapter serving: hot-swap LoRA adapters per request, manage adapter cache alongside KV cache
- Graceful model eviction: when loading a new model exceeds memory, evict least-recently-used model's KV cache first, then unload model weights if needed

### Boundaries
- All models share a single server process (no multi-process model isolation)
- No cross-model KV cache sharing (prefix cache is per-model)
- No automatic model selection / routing intelligence (client specifies model explicitly)
- LoRA adapters must be compatible with the base model (no runtime validation of adapter-model compatibility beyond shape checks)

### Verification
- [ ] Load 2 models concurrently (e.g., Gemma 4 E4B + LLaMA 3.2 1B), route requests to each by model name
- [ ] Concurrent requests to different models produce correct output (no cross-contamination)
- [ ] Memory budgets enforced: loading a model beyond budget fails with clear error
- [ ] Model unload frees all associated memory (KV blocks, weights, LoRA adapters)
- [ ] LoRA hot-swap: same base model serves 2 different LoRA adapters in consecutive requests
- [ ] `/v1/models` endpoint lists all loaded models with status and memory usage
- [ ] Benchmark: multi-model overhead < 5% vs single-model serving for same request load

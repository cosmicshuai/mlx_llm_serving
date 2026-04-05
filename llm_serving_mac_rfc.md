# RFC: High-Throughput, Low-Latency LLM Serving on Apple Silicon

**Status:** Draft
**Author:** Shuai Wang
**Created:** 2026-04-04
**Last Updated:** 2026-04-04

---

## Overview

Build a high-throughput, low-latency LLM inference serving system optimized for Apple Silicon Macs, using MLX as the compute backend and C++ as the primary language.

---

## Goals

1. Support transformer models hosted on HuggingFace
2. Implement modern inference optimizations (paged attention, continuous batching, speculative decoding, etc.)
3. Run natively on Apple Silicon Macs using MLX (Metal GPU acceleration)

---

## Functional Requirements

### FR1 — Model Loading

- Load transformer models from HuggingFace Hub (safetensors format)
- Load the tokenizer assets, model config, and chat template metadata required by each supported checkpoint family
- Support common architectures: Gemma 4, LLaMA, Mistral, Phi, Qwen (Gemma 4 as primary, text-only mode)
- Support quantized weight formats (GGUF, AWQ, GPTQ) as import/load formats, plus on-the-fly quantization (W4A16, W8A8)
- Normalize imported checkpoints into a canonical internal runtime representation instead of maintaining separate execution paths per on-disk format
- Hot-reload models without server restart; v1 scope is atomic replacement of the currently active model within one server process

### FR2 — Inference API

- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`)
- Streaming responses via SSE (Server-Sent Events)
- Support chat and completion modes
- Configurable sampling: temperature, top-k, top-p, repetition penalty, stop sequences, and max tokens

### FR3 — Inference Optimizations

- KV cache with paged memory management
- Paged attention
- Continuous batching (iteration-level scheduling)
- Speculative decoding with configurable draft model
- Prefix caching (shared system prompt KV reuse)

### FR4 — Request Management

- Request queue with priority levels
- Preemption (swap/recompute) when memory is exhausted
- Per-request timeout and cancellation
- Token budget limits per request

### FR5 — Multi-Model Serving (Extension, post-v1)

- Serve multiple models concurrently from a single server instance
- Per-request model selection via API parameter (e.g., `"model": "gemma-4-e4b"`)
- Shared memory pool across models with per-model memory budgets
- Model lifecycle management: load, unload, and swap models at runtime
- Dynamic LoRA adapter serving: hot-swap LoRA adapters per request, manage adapter cache alongside KV cache

---

## Non-Functional Requirements

### NFR1 — Performance

| Metric                              | Target                                                   |
| ----------------------------------- | -------------------------------------------------------- |
| Time to first token (TTFT)          | < 200ms for 1K prompt tokens                             |
| Decode throughput (single request)  | > 30 tokens/sec (Gemma 4 E4B, 8B params)                 |
| Decode throughput (batched)         | > 200 tokens/sec aggregate (E4B, 8+ concurrent requests) |
| Memory utilization                  | > 90% of available unified memory usable for KV cache    |

- Targets are measured on a pinned primary benchmark machine with a fixed prompt suite, model revision, tokenizer assets, and concurrency mix.
- v1 sign-off requires demonstrating these absolute targets or explicitly amending the RFC with the measured gap and rationale.

### NFR2 — Platform

- Apple Silicon only (M1/M2/M3/M4 family)
- MLX as compute backend (Metal GPU acceleration)
- macOS 14+ (Sonoma)
- C++20, built with CMake

### NFR3 — Reliability

- Graceful degradation under memory pressure (evict, don't crash)
- Request-level isolation (one bad input doesn't take down the server)
- Health check endpoint (`/health`, `/ready`)
- Structured logging with request correlation IDs

### NFR4 — Observability

- Prometheus-compatible metrics endpoint (`/metrics`)
- Key metrics: TTFT, decode latency p50/p95/p99, batch size distribution, KV cache hit rate, memory usage, queue depth
- Request-level tracing

### NFR5 — Extensibility

- Pluggable model architecture support (register new arch without modifying core)
- Pluggable scheduling policies
- Pluggable quantization backends
- Pluggable tokenizer, chat template, and checkpoint loader interfaces

---

## Technical Decisions

| Decision          | Choice   | Rationale                                                        |
| ----------------- | -------- | ---------------------------------------------------------------- |
| Language          | C++20    | First-class MLX API, industry standard for inference, Meta stack |
| Compute backend   | MLX      | Native Apple Silicon / Metal support, clean C++ API              |
| Build system      | CMake    | MLX uses CMake, natural fit                                      |
| Model format      | safetensors | HuggingFace standard, memory-mappable, safe                  |
| Quantized import  | Canonical internal runtime layout | Support GGUF/AWQ/GPTQ without carrying separate execution backends |
| Tokenization / templates | Pluggable loaders and renderers | Required to support multiple checkpoint families cleanly |
| API compatibility | OpenAI   | Industry standard, broad client ecosystem                        |

---

## Out of Scope (v1)

- Multi-node / distributed inference
- Training or fine-tuning
- Vision / multimodal models
- Tensor parallelism across devices
- Authentication / rate limiting (delegate to reverse proxy)

---

## Implementation Roadmap

| Phase | Focus                        | Key Deliverables                                       |
| ----- | ---------------------------- | ------------------------------------------------------ |
| 0     | Component Design             | Architecture diagram, interfaces, stubs                |
| 1     | MVP (Gemma 4 E4B text-only)  | Naive e2e: load model → prompt in → tokens out         |
| 2     | KV Cache                     | Block-based paged memory manager                       |
| 3     | Paged Attention              | Metal kernel for non-contiguous KV access              |
| 4     | Quantization                 | W4A16, W8A8 quantization support                       |
| 5     | Batched Prefill              | Multi-sequence prefill in single forward pass, request-scoped decode controls |
| 6     | Continuous Batching          | Iteration-level scheduler, request queue               |
| 7     | Speculative Decoding         | Draft model integration, token verification            |
| 8     | Prefix Caching               | Hash-based KV block sharing across requests            |
| 9     | Serving Layer & Observability| HTTP API, metrics, structured logging, single-model hot-reload, release-gate report |
| 10    | Multi-Model Serving (Ext)    | Model registry, per-request routing, LoRA hot-swap     |

---

## References

- [vLLM: Efficient Memory Management for LLM Serving (Kwon et al.)](https://arxiv.org/abs/2309.06180)
- [MLX Framework (ml-explore/mlx)](https://github.com/ml-explore/mlx)
- [FlashAttention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- [SpecInfer (Miao et al.)](https://arxiv.org/abs/2305.09781)
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)

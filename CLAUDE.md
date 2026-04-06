# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

High-throughput LLM inference server for Apple Silicon, written in C++20 with MLX as the compute backend. Currently in early development (Phase 0+1) — the binary builds and links all dependencies but does not yet load or run models. Target model is Gemma 4 E4B (text-only).

## Build Commands

```bash
# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(sysctl -n hw.ncpu)

# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel $(sysctl -n hw.ncpu)

# Run all tests
cd build && ctest --output-on-failure

# Run a specific test
./build/llm_serve_tests --gtest_filter="SmokeTest.MlxArrayCreation"

# List all tests
./build/llm_serve_tests --gtest_list_tests

# After fresh clone — initialize tokenizers-cpp submodule
git submodule update --init --recursive
```

First build takes several minutes (CMake FetchContent fetches all deps, Rust tokenizer compiles from source). Subsequent builds are incremental.

## Architecture

```
CLI (main.cpp + CLI11) → Engine → Model (MLX ops) → Metal GPU
                          ↑
        Config parser, weight loader, tokenizer feed in
        KV cache is internal state
        Sampling is output stage
```

**Component layout in `src/`:**
- `core/` — types, interfaces, configs (inter-component contracts)
- `model/` — model graph, layers, forward pass (MLX C++ API)
- `tokenizer/` — wraps `mlc-ai/tokenizers-cpp` (Rust FFI → HF tokenizers)
- `loader/` — safetensors parser, weight mapping (memory-mapped, bf16)
- `sampling/` — sampling strategies (greedy, temperature, top-k/p)
- `serving/` — HTTP server (future Phase 9)
- `scheduler/` — request scheduler (future Phase 6)

All subdirectories under `src/` are currently empty scaffolds except `main.cpp`.

## Key Dependencies

All fetched automatically via CMake FetchContent except tokenizers-cpp (git submodule requiring Rust toolchain).

| Library | Integration | Purpose |
|---------|-------------|---------|
| MLX v0.31.1 | FetchContent | Apple Silicon ML compute (Metal backend) |
| tokenizers-cpp | git submodule | HF tokenizers Rust FFI (loads `tokenizer.json`) |
| minja | FetchContent | Chat template rendering (Jinja subset) |
| nlohmann/json v3.12.0 | FetchContent | JSON parsing |
| spdlog v1.15.3 | FetchContent | Structured logging |
| CLI11 v2.4.2 | FetchContent | CLI argument parsing |
| GoogleTest v1.15.2 | FetchContent | Testing framework |

**Critical:** `nlohmann/json` MUST be declared first in CMakeLists.txt. Both MLX and minja internally FetchContent-declare their own json versions — CMake's "first declaration wins" rule ensures a single consistent copy. Do not reorder the FetchContent blocks.

## CMake Targets

- `llm-serve` — main executable (links all deps)
- `llm_serve_tests` — GoogleTest suite (currently smoke tests only)

New source files must be added to the relevant target in `CMakeLists.txt`. New test files go into `llm_serve_tests` or a new test executable registered with `gtest_discover_tests()`.

## Model Spec

The authoritative Gemma 4 E4B architecture specification is at `docs/models/gemma4_e4b.md`. Key details:
- 42-layer decoder-only transformer, hybrid sliding window (512 tokens) + global attention
- Dual RoPE: standard (theta=10k) for sliding layers, proportional (theta=1M, 25% partial) for global layers
- Shared KV cache: layers 24–41 reuse KV from earlier layers (no own K/V projections)
- Per-Layer Embeddings (PLE): separate 256-d embedding per layer
- GQA 4:1 (8 Q heads, 2 KV heads), head_dim=256 (sliding) / 512 (global)
- Logit softcapping: `30.0 * tanh(logits / 30.0)`
- Tokenizer is HF BPE format (`tokenizer.json`), NOT SentencePiece

## Task Tracking

Current phase tasks are tracked in `docs/phase_0_1_tasks.md`. The full multi-phase implementation plan is in `plan.md`.

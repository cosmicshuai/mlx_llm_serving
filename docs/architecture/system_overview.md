# System Architecture Overview

## Component Diagram

```mermaid
flowchart TD
    subgraph Ingress["Request Ingress"]
        CLI["CLI\n(main.cpp + CLI11)"]
        HTTP["HTTP API\n(serving/)"]
    end

    subgraph Init["Startup — LLMEngine.initialize()"]
        CONFIG["Config Parser\n(config.json → TextModelConfig)"]
        WLOADER["Weight Loader\n(safetensors, mmap, bf16)"]
        TOK_LOAD["Tokenizer Loader\n(tokenizer.json via tokenizers-cpp)"]
    end

    subgraph EngineBox["LLMEngine — step() loop"]
        direction TB
        SCHED["Scheduler"]
        MR["ModelRunner"]
        SAMPLER["Sampler"]
    end

    subgraph ModelBox["Model"]
        direction TB
        LAYERS["Transformer Blocks\n(Attention + MLP + RMSNorm + RoPE + PLE)"]
    end

    KV["KVCacheManager"]
    TOKENIZER["Tokenizer\n(encode + chat template)"]
    DETOK["Detokenizer\n(incremental decode, streaming)"]
    MLX["MLX Runtime\n(Metal kernels, eval)"]
    PCACHE["Prefix Cache"]

    %% Request flow
    CLI --> TOKENIZER
    HTTP -.-> TOKENIZER
    TOKENIZER --> SCHED
    SCHED -- "select sequences\nfor prefill/decode" --> MR
    MR -- "forward pass" --> LAYERS
    LAYERS --> MLX
    MR -- "logits" --> SAMPLER
    SAMPLER -- "token IDs" --> DETOK
    DETOK --> CLI
    DETOK -.-> HTTP

    %% KV cache interactions
    SCHED -- "alloc/evict blocks" --> KV
    MR -- "read/write KV tensors" --> KV
    PCACHE -.-> KV

    %% Initialization flow
    CONFIG --> EngineBox
    WLOADER --> ModelBox
    TOK_LOAD --> TOKENIZER
    TOK_LOAD --> DETOK

    %% Phase 0+1 (building now) — yellow
    style CLI fill:#ff9,stroke:#333,color:#000
    style CONFIG fill:#ff9,stroke:#333,color:#000
    style WLOADER fill:#ff9,stroke:#333,color:#000
    style TOK_LOAD fill:#ff9,stroke:#333,color:#000
    style TOKENIZER fill:#ff9,stroke:#333,color:#000
    style DETOK fill:#ff9,stroke:#333,color:#000
    style SCHED fill:#ff9,stroke:#333,color:#000
    style MR fill:#ff9,stroke:#333,color:#000
    style SAMPLER fill:#ff9,stroke:#333,color:#000
    style LAYERS fill:#ff9,stroke:#333,color:#000
    style KV fill:#ff9,stroke:#333,color:#000
    style MLX fill:#ff9,stroke:#333,color:#000
    style EngineBox fill:#ffffcc,stroke:#333,color:#000
    style ModelBox fill:#ffffcc,stroke:#333,color:#000
    style Init fill:#ffffcc,stroke:#333,color:#000

    %% Future phases — gray dashed
    style HTTP fill:#ddd,stroke:#999,color:#666,stroke-dasharray: 5 5
    style PCACHE fill:#ddd,stroke:#999,color:#666,stroke-dasharray: 5 5
    style Ingress fill:transparent,stroke:#999,color:#000
```

## Legend

| Visual | Meaning |
|--------|---------|
| **Yellow** solid | Phase 0+1 — building now |
| **Gray** dashed | Future phases (Phase 2+) |

## LLMEngine Lifecycle

```
initialize()    Load config, weights, tokenizer. Build model graph. Allocate KV cache.
     │
     ▼
  step()  ◄──── main loop, called repeatedly until all requests complete
     │
     ├─ 1. scheduler.schedule()        → pick sequence(s) to run (prefill or decode)
     ├─ 2. model_runner.execute()      → forward pass through model, read/write KV cache
     ├─ 3. sampler.sample(logits)      → pick next token(s)
     ├─ 4. check stopping conditions   → EOS, max_tokens, stop sequences
     ├─ 5. detokenizer.decode(token)   → incremental text → stream to client
     └─ 6. return results              → finished sequences, streaming tokens
     │
     ▼
 shutdown()     Free KV cache, unload weights, release Metal resources.
```

## Component → Source Mapping

| Component | Source Location | Phase |
|-----------|---------------|-------|
| CLI | `src/main.cpp` | 1 |
| HTTP API | `src/serving/` | 9 |
| Config Parser | `src/core/` or `src/loader/` | 1 |
| Weight Loader | `src/loader/` | 1 |
| Tokenizer / Detokenizer | `src/tokenizer/` | 1 |
| LLMEngine | `src/core/` | 1 |
| Scheduler | `src/scheduler/` | 1 (SimpleScheduler), 6 (ContinuousBatchingScheduler) |
| ModelRunner | `src/model/` | 1 |
| Model / Layers | `src/model/` | 1 |
| KVCacheManager | `src/core/` | 1 (NaiveKVCacheManager), 2–3 (PagedKVCacheManager) |
| Sampler | `src/sampling/` | 1 |
| Prefix Cache | `src/core/` | 8 |
| MLX Runtime | external (FetchContent) | — |

## Interface Hierarchy (Phase 0+1 stubs → later implementations)

```
Scheduler (base)
├── SimpleScheduler              — Phase 1: single request, always run
└── ContinuousBatchingScheduler  — Phase 6: admission, batching, preemption, priority

KVCacheManager (base)
├── NaiveKVCacheManager          — Phase 1: contiguous pre-allocated buffer
└── PagedKVCacheManager          — Phase 2–3: block allocator, paged memory

ModelRunner (base)
└── GemmaModelRunner             — Phase 1: Gemma 4 forward pass
```

## Data Flow Summary

1. **Request in:** CLI (or HTTP) → chat template renders messages → tokenizer encodes to token IDs
2. **Scheduling:** Scheduler picks which sequence(s) to run, allocates KV cache blocks
3. **Execution:** ModelRunner runs prefill or decode step through transformer layers via MLX Metal kernels
4. **Sampling:** Engine passes logits to Sampler, gets next token ID(s)
5. **Output:** Detokenizer incrementally decodes token IDs → streams text back to client
6. **Loop:** Repeat steps 2–5 until EOS, max_tokens, or stop sequence

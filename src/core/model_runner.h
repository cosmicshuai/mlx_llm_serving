#pragma once

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

#include "src/core/kv_cache_manager.h"

namespace llm_serve {

/// Abstract model runner.
///
/// Executes the transformer forward pass. Implementations are
/// architecture-specific (e.g., GemmaModelRunner). The engine calls
/// prefill() once for the prompt, then decode_step() in a loop.
class ModelRunner {
   public:
    virtual ~ModelRunner() = default;

    /// Run the full prompt through the model in one forward pass.
    /// Populates the KV cache for all prompt positions.
    ///
    /// @param token_ids  Prompt token IDs (length = prompt length).
    /// @param kv_cache   KV cache to populate (mutated).
    /// @return Logits for the last position, shape [vocab_size].
    virtual mlx::core::array prefill(
        const std::vector<int32_t>& token_ids,
        KVCacheManager& kv_cache) = 0;

    /// Run a single decode step for one new token.
    /// Appends the new K/V to the cache.
    ///
    /// @param token_id   The previously sampled token.
    /// @param kv_cache   KV cache to read from and append to (mutated).
    /// @return Logits for the next position, shape [vocab_size].
    virtual mlx::core::array decode_step(
        int32_t token_id,
        KVCacheManager& kv_cache) = 0;
};

}  // namespace llm_serve

#pragma once

#include <cstdint>
#include <utility>

#include <mlx/mlx.h>

namespace llm_serve {

/// Abstract KV cache manager.
///
/// Stores Key and Value tensors per layer across decode steps. The model
/// appends new K/V each step; attention reads all cached K/V for its layer.
/// Shared KV layers read from their source layer via get() and never append.
///
/// Phase 1: naive contiguous pre-allocated buffer.
/// Phase 2+: paged block allocator with eviction (extends this interface).
class KVCacheManager {
   public:
    virtual ~KVCacheManager() = default;

    /// Get cached K and V tensors for a layer.
    ///
    /// @param layer_idx  Layer index (0-based). Shared KV layers pass their
    ///                   source layer index to read from the right cache slot.
    /// @return {K, V} tensors of shape [seq_len, num_kv_heads, head_dim].
    ///         Empty arrays if nothing cached yet.
    virtual std::pair<mlx::core::array, mlx::core::array> get(
        int32_t layer_idx) const = 0;

    /// Append new K/V tensors for a layer at the next sequence position.
    ///
    /// @param layer_idx  Layer index (must be a layer that produces K/V,
    ///                   not a shared KV layer).
    /// @param k  Key tensor for the new position(s), shape [num_new, num_kv_heads, head_dim].
    /// @param v  Value tensor, same shape as k.
    virtual void append(int32_t layer_idx,
                        const mlx::core::array& k,
                        const mlx::core::array& v) = 0;

    /// Number of cached positions (tokens) across all layers.
    virtual int32_t seq_len() const = 0;

    /// Clear all cached state. Called between sequences.
    virtual void reset() = 0;
};

}  // namespace llm_serve

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace llm_serve {

/// Configuration for token sampling during generation.
///
/// Applied per-request. Controls how the model's raw logit distribution
/// is transformed into a token selection. Pipeline order:
/// softcap → temperature scale → softmax → top_k → top_p → sample.
struct SamplingConfig {
    /// Scales logits before softmax. Lower sharpens (→ greedy), higher flattens.
    /// 0.0 = greedy (argmax). Default from Gemma 4 generation_config.
    float temperature = 1.0f;

    /// Keep only the top-k highest-probability tokens after softmax.
    /// 0 = disabled (no top-k filtering).
    int32_t top_k = 64;

    /// Keep the smallest token set whose cumulative probability >= top_p.
    /// 1.0 = disabled (no nucleus filtering).
    float top_p = 0.95f;

    /// Maximum tokens to generate. Generation stops regardless of EOS.
    int32_t max_tokens = 512;

    /// Token IDs that trigger end of generation (in addition to model's default EOS).
    std::vector<int32_t> stop_token_ids;

    /// RNG seed for reproducible sampling. std::nullopt = non-deterministic.
    std::optional<uint64_t> seed;

    // TODO(phase-9): repetition_penalty — penalize repeated tokens in HTTP API
    // TODO(phase-9): stop_sequences (string-level) — stop on substring match, not just token ID
};

}  // namespace llm_serve

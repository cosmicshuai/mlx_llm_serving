#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm_serve {

/// Why generation stopped.
enum class FinishReason : uint8_t {
    kMaxTokens,   ///< Hit max_tokens limit
    kEOS,         ///< Model emitted EOS token
    kStopToken,   ///< Hit a user-specified stop token ID
};

/// Timing statistics for a single generation request.
struct TimingStats {
    double ttft_ms = 0.0;            ///< Time to first token (prefill latency)
    double decode_ms = 0.0;          ///< Total decode phase wall time
    double total_ms = 0.0;           ///< Wall clock from request start to last token
    int32_t prompt_tokens = 0;       ///< Number of input tokens after tokenization
    int32_t generated_tokens = 0;    ///< Number of tokens generated
    double decode_tokens_per_sec = 0.0;  ///< generated_tokens / decode_ms * 1000
    size_t peak_memory_bytes = 0;    ///< GPU peak memory during this request
};

/// Result returned by the engine after a generation request completes.
struct GenerationResult {
    std::vector<int32_t> token_ids;  ///< Generated token IDs (excludes prompt)
    std::string text;                ///< Decoded text from generated tokens
    FinishReason finish_reason = FinishReason::kEOS;
    TimingStats timing;
};

}  // namespace llm_serve

#pragma once

#include <cstdint>

#include <mlx/mlx.h>

#include "src/core/sampling_config.h"

namespace llm_serve {

/// Result of sampling a single token from logits.
struct SampleResult {
    int32_t token_id = 0;
    float log_prob = 0.0f;  ///< Log-probability of the selected token
};

/// Abstract token sampler.
///
/// Constructed with a SamplingConfig (bound per request). Called each decode
/// step to select the next token from the model's logit distribution.
/// Pipeline: softcap → temperature → softmax → top_k → top_p → sample.
class Sampler {
   public:
    virtual ~Sampler() = default;

    /// Sample a token from the logit distribution.
    ///
    /// @param logits  Raw logits from the model, shape [vocab_size].
    /// @return Selected token ID and its log-probability.
    virtual SampleResult sample(const mlx::core::array& logits) = 0;
};

}  // namespace llm_serve

#pragma once

#include <cstdint>
#include <vector>

#include "src/core/sampling_config.h"

namespace llm_serve {

/// Whether a sequence needs prefill or decode this step.
enum class ScheduleAction : uint8_t {
    kPrefill,  ///< First forward pass — process full prompt
    kDecode,   ///< Subsequent steps — one token at a time
};

/// A single sequence scheduled for execution this step.
struct ScheduledSequence {
    int64_t request_id = 0;
    ScheduleAction action = ScheduleAction::kPrefill;

    /// Prefill: full prompt token IDs. Decode: single previously-sampled token.
    std::vector<int32_t> token_ids;
};

/// Batch of sequences to process in one engine step.
struct ScheduleBatch {
    std::vector<ScheduledSequence> sequences;
};

/// Abstract request scheduler.
///
/// Manages the queue of generation requests and decides which sequences
/// to execute each engine step. The engine calls schedule() in a loop.
///
/// Phase 1 (SimpleScheduler): single request, always returns batch of 1.
/// Phase 6 (ContinuousBatchingScheduler): multi-request, priority-based
/// admission, preemption, dynamic batch composition.
class Scheduler {
   public:
    virtual ~Scheduler() = default;

    /// Enqueue a new generation request.
    ///
    /// @param request_id    Unique ID for this request.
    /// @param prompt_tokens Tokenized prompt.
    /// @param config        Per-request sampling configuration.
    virtual void add_request(int64_t request_id,
                             std::vector<int32_t> prompt_tokens,
                             SamplingConfig config) = 0;

    /// Decide which sequences to run this step.
    /// Returns an empty batch if no work is ready.
    virtual ScheduleBatch schedule() = 0;

    /// Mark a request as finished (completed, cancelled, or timed out).
    /// Frees any scheduler-internal state for this request.
    virtual void finish_request(int64_t request_id) = 0;

    /// True if there are pending or active requests.
    virtual bool has_pending() const = 0;
};

}  // namespace llm_serve

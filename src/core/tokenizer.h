#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace llm_serve {

/// Abstract tokenizer interface.
///
/// Encodes text to token IDs and decodes back. Implementations wrap a
/// concrete backend (e.g., tokenizers-cpp for HF BPE, SentencePiece).
/// The engine and model code depend only on this interface.
class Tokenizer {
   public:
    virtual ~Tokenizer() = default;

    /// Encode text into token IDs.
    /// Does NOT prepend BOS or append EOS — caller decides.
    virtual std::vector<int32_t> encode(std::string_view text) const = 0;

    /// Decode token IDs back to text.
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    /// Total vocabulary size (e.g., 262144 for Gemma 4).
    virtual int32_t vocab_size() const = 0;

    /// Beginning-of-sequence token ID.
    virtual int32_t bos_id() const = 0;

    /// End-of-sequence token ID.
    virtual int32_t eos_id() const = 0;
};

}  // namespace llm_serve

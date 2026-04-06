#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace llm_serve {

/// A single message in a chat conversation.
struct ChatMessage {
    std::string role;     ///< "system", "user", "assistant", "tool"
    std::string content;  ///< Message text (treated as data, never as control tokens)
};

/// Abstract chat template renderer.
///
/// Converts structured conversation turns into the model-specific prompt
/// format. Implementations load a Jinja template (e.g., from
/// tokenizer_config.json) and evaluate it via minja or equivalent.
///
/// The renderer is the only component that produces structural tags.
/// User-supplied content is interpolated as data to prevent prompt injection.
class ChatTemplateRenderer {
   public:
    virtual ~ChatTemplateRenderer() = default;

    /// Render a conversation into the model's expected prompt format.
    ///
    /// @param messages       Ordered conversation turns.
    /// @param add_generation_prompt  If true, append the model's "start generating"
    ///                               suffix (e.g., "<|turn|>model\n" for Gemma 4).
    /// @return Formatted prompt string ready for tokenization.
    virtual std::string render(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true) const = 0;
};

}  // namespace llm_serve

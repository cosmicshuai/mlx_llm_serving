#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm_serve {

/// Attention layer type within a transformer model.
enum class AttentionLayerType : uint8_t {
    kSlidingWindow,  ///< Local attention within a fixed window
    kGlobal,         ///< Full causal attention over all tokens
};

/// RoPE (Rotary Position Embedding) variant.
enum class RoPEType : uint8_t {
    kDefault,       ///< Standard RoPE
    kProportional,  ///< Proportional RoPE (scaled theta)
};

/// Activation function for the MLP feed-forward layer.
enum class ActivationType : uint8_t {
    kGELU,             ///< Standard GELU
    kGELUPytorchTanh,  ///< GELU with tanh approximation (PyTorch-compatible)
    kSiLU,             ///< SiLU / Swish (LLaMA, Mistral)
    kReLU,             ///< ReLU
};

/// Data type for model weights and computation.
enum class DType : uint8_t {
    kFloat32,
    kFloat16,
    kBFloat16,
};

/// RoPE configuration for a specific attention type.
struct RoPEConfig {
    RoPEType type = RoPEType::kDefault;

    /// Base frequency for position encoding.
    float theta = 10000.0f;

    /// Fraction of head dimension to apply rotary embeddings to.
    /// 1.0 = full head dim, 0.25 = 25% of head dim (proportional RoPE).
    float partial_rotary_factor = 1.0f;
};

/// Per-layer embedding (PLE) configuration.
/// Optional — only architectures like Gemma 4 use a separate low-dimensional
/// embedding per layer to modulate the hidden state at each layer's input.
struct PLEConfig {
    int32_t hidden_size = 0;  ///< PLE embedding dim (e.g., 256)
    int32_t vocab_size = 0;   ///< PLE vocab size (typically matches main vocab)
};

/// Attention configuration for a specific layer type.
/// Each distinct AttentionLayerType maps to one of these.
struct AttentionConfig {
    int32_t head_dim = 0;       ///< Dimension per attention head
    RoPEConfig rope;            ///< Positional encoding config
    int32_t sliding_window = 0; ///< Window size (0 = no windowing / full attention)
};

/// Text model architecture configuration.
///
/// Parsed from a HuggingFace model's config.json. Designed to represent
/// multiple transformer architectures (Gemma, LLaMA, Mistral, Phi, Qwen)
/// without architecture-specific fields in the struct definition.
struct TextModelConfig {
    // ── Identity ────────────────────────────────────────────────────────────
    std::string model_type;  ///< e.g., "gemma4_text", "llama", "mistral"

    // ── Dimensions ──────────────────────────────────────────────────────────
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;   ///< MLP intermediate (feed-forward) dim
    int32_t num_hidden_layers = 0;
    int32_t max_position_embeddings = 0;

    // ── Attention (global) ──────────────────────────────────────────────────
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;  ///< For GQA; equals num_attention_heads for MHA
    bool attention_bias = false;      ///< Whether Q/K/V/O projections use bias

    // ── Layer pattern ───────────────────────────────────────────────────────
    /// Attention type for each layer (length == num_hidden_layers).
    /// Determines which AttentionConfig applies per layer.
    std::vector<AttentionLayerType> layer_types;

    /// Config for sliding window attention layers.
    AttentionConfig sliding_attention;
    /// Config for global (full) attention layers.
    AttentionConfig global_attention;

    // ── Shared KV cache ─────────────────────────────────────────────────────
    /// Number of layers (counting from the last) that reuse K/V tensors from
    /// earlier layers instead of computing their own. 0 = no sharing.
    /// Source layer for shared layer i: i - num_kv_shared_layers.
    int32_t num_kv_shared_layers = 0;

    // ── Normalization ───────────────────────────────────────────────────────
    float rms_norm_eps = 1e-6f;

    // ── Activation ──────────────────────────────────────────────────────────
    ActivationType hidden_activation = ActivationType::kGELUPytorchTanh;

    // ── Embeddings ──────────────────────────────────────────────────────────
    bool tie_word_embeddings = false;

    /// Per-layer embedding config. std::nullopt if the architecture has no PLE.
    std::optional<PLEConfig> ple;

    // ── Output ──────────────────────────────────────────────────────────────
    /// Logit softcapping: cap * tanh(logits / cap). 0.0 = disabled.
    float final_logit_softcapping = 0.0f;

    // ── Precision ───────────────────────────────────────────────────────────
    DType dtype = DType::kBFloat16;
};

}  // namespace llm_serve

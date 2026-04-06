#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

#include "src/core/text_model_config.h"

namespace llm_serve {

/// Named collection of model weight tensors.
/// Key: parameter path (e.g., "model.layers.0.self_attn.q_proj.weight").
/// Value: MLX array (dtype, shape, data — possibly memory-mapped).
using ModelWeights = std::unordered_map<std::string, mlx::core::array>;

/// Abstract checkpoint loader.
///
/// Loads model weights from disk into MLX arrays. Implementations handle
/// a specific on-disk format (safetensors, GGUF, etc.) and map external
/// parameter names to internal names.
class CheckpointLoader {
   public:
    virtual ~CheckpointLoader() = default;

    /// Load model weights from a checkpoint directory or file.
    ///
    /// @param path    Path to the model directory (e.g., HuggingFace repo clone)
    ///                or a specific checkpoint file.
    /// @param config  Model config — used to validate tensor shapes and filter
    ///                irrelevant keys (e.g., skip vision/audio tensors).
    /// @return Map of parameter name → MLX array.
    /// @throws std::runtime_error on I/O errors, missing tensors, or shape mismatches.
    virtual ModelWeights load(
        const std::filesystem::path& path,
        const TextModelConfig& config) const = 0;
};

}  // namespace llm_serve

# llm-serve

High-throughput, low-latency LLM inference server for Apple Silicon, built with MLX and C++20.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14.0+ (Sonoma)
- Xcode with Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- CMake 3.25+ (`brew install cmake`)
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

## Quick Start

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(sysctl -n hw.ncpu)
```

### Run

```bash
# Basic generation (coming soon — model loading not yet implemented)
./build/llm-serve --model /path/to/model --prompt "Hello, world!"

# Interactive chat mode (coming soon)
./build/llm-serve --model /path/to/model --chat
```

### Download a Model

This server loads HuggingFace models in safetensors format. To download a model:

```bash
# Install huggingface-cli if you don't have it
pip install huggingface_hub

# Download Gemma 4 E4B (primary supported model, ~16 GB)
huggingface-cli download google/gemma-4-E4B --local-dir ./models/gemma-4-E4B
```

The model directory should contain at minimum:
- `config.json` — model architecture config
- `model.safetensors` — model weights
- `tokenizer.json` — tokenizer vocabulary and merges
- `tokenizer_config.json` — tokenizer metadata and chat template

### CLI Options (planned)

| Flag | Description | Default |
|------|-------------|---------|
| `--model <path>` | Path to HuggingFace model directory | (required) |
| `--prompt <text>` | Raw prompt (bypasses chat template) | |
| `--chat` | Interactive multi-turn chat mode | |
| `--temperature <float>` | Sampling temperature | 1.0 |
| `--top-k <int>` | Top-k sampling | 64 |
| `--top-p <float>` | Nucleus sampling threshold | 0.95 |
| `--max-tokens <int>` | Maximum tokens to generate | 512 |
| `--seed <int>` | Random seed for reproducibility | |

## Project Status

This project is in early development (Phase 0+1). The binary currently builds and links all dependencies but does not yet load or run models. See [`docs/phase_0_1_tasks.md`](docs/phase_0_1_tasks.md) for the current task list.

## License

See [LICENSE](LICENSE) for details.

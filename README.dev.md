# Developer Guide

## Prerequisites

| Tool | Minimum Version | Install |
|------|----------------|---------|
| CMake | 3.25 | `brew install cmake` |
| Xcode CLI tools | 15.0+ | `xcode-select --install` |
| Metal Toolchain | — | `xcodebuild -downloadComponent MetalToolchain` |
| Rust | 1.70+ | `rustup update stable` |

If this is a fresh Xcode install, also run:
```bash
xcodebuild -runFirstLaunch
```

## Building

```bash
# Release build (optimized)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(sysctl -n hw.ncpu)

# Debug build (symbols, assertions)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel $(sysctl -n hw.ncpu)
```

First build takes several minutes — CMake fetches all dependencies via FetchContent, and the Rust tokenizer library compiles from source. Subsequent builds are incremental.

### IDE Setup

The build generates `build/compile_commands.json` for clangd-based editors (VS Code, CLion, Neovim). Symlink it to the project root if your editor expects it there:

```bash
ln -sf build/compile_commands.json .
```

## Running Tests

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run a specific test
./build/llm_serve_tests --gtest_filter="SmokeTest.MlxArrayCreation"

# List available tests
./build/llm_serve_tests --gtest_list_tests
```

## Project Structure

```
src/
  main.cpp            # CLI entry point
  core/               # Types, interfaces, configs
  model/              # Model graph, layers, forward pass
  tokenizer/          # Tokenizer wrapper
  loader/             # Safetensors parser, weight mapping
  sampling/           # Sampling strategies
  serving/            # HTTP server (Phase 9)
  scheduler/          # Request scheduler (Phase 6)
include/              # Public headers
tests/                # GoogleTest suites
third_party/
  tokenizers-cpp/     # HF tokenizers Rust FFI (git submodule)
docs/
  architecture/       # System diagrams
  models/             # Model specifications
benchmarks/           # Benchmark harness and results
```

## Dependencies

All dependencies are fetched automatically at configure time. No manual installation needed beyond the prerequisites above.

| Library | Version | Purpose | Integration |
|---------|---------|---------|-------------|
| [MLX](https://github.com/ml-explore/mlx) | v0.31.1 | Apple Silicon ML compute | FetchContent |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) | latest | HF tokenizers Rust FFI | git submodule |
| [minja](https://github.com/google/minja) | latest | Chat template rendering | FetchContent |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.12.0 | JSON parsing | FetchContent |
| [spdlog](https://github.com/gabime/spdlog) | v1.15.3 | Structured logging | FetchContent |
| [CLI11](https://github.com/CLIUtils/CLI11) | v2.4.2 | Argument parsing | FetchContent |
| [GoogleTest](https://github.com/google/googletest) | v1.15.2 | Testing | FetchContent |

### Dependency Ordering Note

Both MLX and minja internally fetch `nlohmann/json` via FetchContent. Our `CMakeLists.txt` declares `json` **first** so all three share a single copy (CMake FetchContent "first declaration wins" rule). Do not reorder the FetchContent blocks.

## Adding a New Source File

1. Add the `.cpp` file under the appropriate `src/` subdirectory
2. Add the corresponding `.hpp` header under `include/` (for public headers) or alongside the `.cpp` (for internal headers)
3. Add the source to the relevant CMake target in `CMakeLists.txt`
4. Add tests in `tests/`

## Adding a New Test File

1. Create `tests/my_test.cpp` with GoogleTest fixtures
2. Add it to the `llm_serve_tests` target sources in `CMakeLists.txt`, or create a new test executable and register it with `gtest_discover_tests()`

## Submodule Management

The `third_party/tokenizers-cpp` directory is a git submodule. After cloning the repo:

```bash
git submodule update --init --recursive
```

## Clean Build

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(sysctl -n hw.ncpu)
```

This re-fetches all FetchContent dependencies and triggers a full Rust rebuild (~3-5 min).

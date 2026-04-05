cmake_minimum_required(VERSION 3.25)
project(llm_serving VERSION 0.1.0 LANGUAGES C CXX)

# ── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── macOS Deployment Target ──────────────────────────────────────────────────
# MLX Metal backend requires macOS >= 14.0
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")

# ── FetchContent Setup ───────────────────────────────────────────────────────
include(FetchContent)

# ── nlohmann/json — MUST be declared FIRST ───────────────────────────────────
# Both MLX (v3.11.3) and minja (v3.12.0) FetchContent_Declare(json ...) internally.
# FetchContent "first declaration wins" — declaring v3.12.0 here ensures
# a single consistent version. v3.12.0 is backward-compatible with v3.11.3.
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# ── spdlog ───────────────────────────────────────────────────────────────────
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spdlog)

# ── CLI11 ────────────────────────────────────────────────────────────────────
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

# ── GoogleTest ───────────────────────────────────────────────────────────────
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# ── MLX ──────────────────────────────────────────────────────────────────────
set(MLX_BUILD_TESTS OFF CACHE INTERNAL "")
set(MLX_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(MLX_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG        v0.31.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(mlx)

# ── minja (header-only, depends on nlohmann/json) ────────────────────────────
set(MINJA_TEST_ENABLED OFF CACHE INTERNAL "")
set(MINJA_EXAMPLE_ENABLED OFF CACHE INTERNAL "")
set(MINJA_FUZZTEST_ENABLED OFF CACHE INTERNAL "")
FetchContent_Declare(
    minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(minja)

# ── tokenizers-cpp (git submodule, requires Rust toolchain) ──────────────────
add_subdirectory(third_party/tokenizers-cpp)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(llm-serve src/main.cpp)

target_link_libraries(llm-serve
    PRIVATE
        mlx
        tokenizers_cpp
        minja
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        CLI11::CLI11
)

target_include_directories(llm-serve
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Testing ──────────────────────────────────────────────────────────────────
enable_testing()
include(GoogleTest)

add_executable(llm_serve_tests tests/smoke_test.cpp)

target_link_libraries(llm_serve_tests
    PRIVATE
        GTest::gtest_main
        mlx
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)

target_include_directories(llm_serve_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

gtest_discover_tests(llm_serve_tests)

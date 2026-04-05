/// @file main.cpp
/// @brief Entry point for llm-serve. Task 1 skeleton: verifies all dependencies link.

#include <cstdlib>
#include <iostream>

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <tokenizers_cpp.h>
#include <minja/minja.hpp>

int main(int argc, char* argv[]) {
    CLI::App app{"llm-serve: LLM inference server for Apple Silicon"};
    CLI11_PARSE(app, argc, argv);

    spdlog::info("llm-serve starting up");

    auto arr = mlx::core::array({1.0f, 2.0f, 3.0f});
    spdlog::info("MLX array created: size={}", arr.size());

    nlohmann::json j = {{"status", "ok"}, {"version", "0.1.0"}};
    spdlog::info("JSON: {}", j.dump());

    spdlog::info("tokenizers-cpp linked successfully");
    spdlog::info("minja linked successfully");

    std::cout << "hello" << std::endl;

    spdlog::info("llm-serve shutting down");
    return EXIT_SUCCESS;
}

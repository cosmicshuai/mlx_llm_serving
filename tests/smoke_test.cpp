/// @file smoke_test.cpp
/// @brief Smoke tests verifying all dependencies are correctly linked.

#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

TEST(SmokeTest, MlxArrayCreation) {
    auto arr = mlx::core::array({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(arr.size(), 3);
}

TEST(SmokeTest, JsonRoundTrip) {
    nlohmann::json j = {{"key", "value"}, {"number", 42}};
    auto parsed = nlohmann::json::parse(j.dump());
    EXPECT_EQ(parsed["key"], "value");
    EXPECT_EQ(parsed["number"], 42);
}

TEST(SmokeTest, SpdlogInit) {
    EXPECT_NO_THROW(spdlog::info("smoke test log message"));
}

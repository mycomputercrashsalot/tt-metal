
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <utility>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class CrossEntropyForwardTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }
};

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<uint32_t>& target) {
    const uint32_t N = target.shape(0);
    const uint32_t C = 1U;
    const uint32_t H = target.shape(1);
    const uint32_t W = 1U;
    xt::xarray<float> target_inputs = xt::zeros<float>({N, C, H, W});

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, 0) = input(n, 0, h, class_index);
        }
    }

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> log_exp_sum_test = xt::log(xt::sum(xt::exp(shifted_input), -1, xt::keep_dims));
    xt::xarray<float> result = -target_inputs + max_input + log_exp_sum_test;
    return result;
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, H = 1;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, H = 2;

    xt::xarray<float> input_tensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 0;
    target_tensor(0, 1) = 2U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

// Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
TEST_F(CrossEntropyForwardTest, DISABLED_CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, NIGHTLY_CrossEntropyForward_Huge_Forward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

namespace {

using ttml::test_utils::two_core_group_seq_len;

struct CrossEntropyFwLaunchTensors {
    ttnn::Tensor input;
    ttnn::Tensor target;
    xt::xarray<float> input_values;  // bf16-rounded, as the device holds them
    xt::xarray<uint32_t> target_values;
};

// Same specs for every call with the same (N, H, W); the values come from `seed`.
CrossEntropyFwLaunchTensors make_inputs(uint32_t N, uint32_t H, uint32_t W, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto target_values =
        test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    auto input = core::from_xtensor(
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, 1U, H, W}, -10.0F, 10.0F, seed), device);
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_values, device, ttnn::Layout::ROW_MAJOR);
    auto input_values = core::to_xtensor(input);
    return {std::move(input), std::move(target), std::move(input_values), std::move(target_values)};
}

void expect_loss_matches(const ttnn::Tensor& result, const CrossEntropyFwLaunchTensors& in, const std::string& launch) {
    const auto expected = calculate_cross_entropy_loss(in.input_values, in.target_values);
    const auto actual = ttml::core::to_xtensor(result);
    ASSERT_EQ(actual.shape(), expected.shape()) << launch;
    EXPECT_TRUE(xt::allclose(actual, expected, 3e-2F, 1e-2F)) << launch;
}

}  // namespace

// A program-cache hit re-patches only the input, target and output addresses. The first launch's tensors stay alive, so
// every buffer of the second launch is at a new address: a stale one reads the first launch's data or overwrites its
// result.
TEST_F(CrossEntropyForwardTest, CrossEntropyForward_CacheHitFreshTensors_TwoCoreGroups) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    device.clear_program_cache();

    const uint32_t N = 2U, H = two_core_group_seq_len(7U), W = 157U;
    auto& rng = autograd::ctx().get_generator();
    const uint32_t seed = rng();

    const auto a = make_inputs(N, H, W, seed);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = metal::cross_entropy_fw(a.input, a.target);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "first launch did not populate the cache";
    expect_loss_matches(result_a, a, "first launch");

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto result_b = metal::cross_entropy_fw(b.input, b.target);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "second launch was not a cache hit";
    EXPECT_NE(b.input.buffer()->address(), a.input.buffer()->address());
    EXPECT_NE(b.target.buffer()->address(), a.target.buffer()->address());
    EXPECT_NE(result_b.buffer()->address(), result_a.buffer()->address());

    expect_loss_matches(result_b, b, "cache hit");
    expect_loss_matches(result_a, a, "first launch, re-read after the cache hit");
}

// The prim takes an optional preallocated output; the public wrapper never passes one. Launch without one, with one,
// with a fresh one and without again, keeping every tensor alive: the last two launches are cache hits whether or not
// the key depends on the output's presence, and each launch must write its own output. Then the other order on an
// empty cache: build the program with a preallocated output, launch without one, then with a fresh one (a hit).
// Preallocated outputs start at -1, which no loss can be.
TEST_F(CrossEntropyForwardTest, CrossEntropyForward_PreallocatedOutput_CacheHit) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    device.clear_program_cache();

    const uint32_t N = 2U, H = two_core_group_seq_len(7U), W = 61U;
    auto& rng = autograd::ctx().get_generator();
    const uint32_t seed = rng();
    const auto make_output = [&]() {
        xt::xarray<float> sentinel = xt::empty<float>({N, 1U, H, 1U});
        sentinel.fill(-1.0F);
        return core::from_xtensor(sentinel, &device);
    };

    const auto a = make_inputs(N, H, W, seed);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = ttnn::prim::ttml_cross_entropy_fw(a.input, a.target, std::nullopt);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "first launch did not populate the cache";

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto output_b = make_output();
    const auto result_b = ttnn::prim::ttml_cross_entropy_fw(b.input, b.target, output_b);

    const auto c = make_inputs(N, H, W, seed + 4U);
    const auto output_c = make_output();
    auto entries_before_hit = device.num_program_cache_entries();
    const auto result_c = ttnn::prim::ttml_cross_entropy_fw(c.input, c.target, output_c);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "preallocated launch was not a cache hit";

    const auto d = make_inputs(N, H, W, seed + 6U);
    entries_before_hit = device.num_program_cache_entries();
    const auto result_d = ttnn::prim::ttml_cross_entropy_fw(d.input, d.target, std::nullopt);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch without output was not a cache hit";

    EXPECT_EQ(result_b.buffer()->address(), output_b.buffer()->address());
    EXPECT_EQ(result_c.buffer()->address(), output_c.buffer()->address());
    EXPECT_NE(result_d.buffer()->address(), result_a.buffer()->address());

    expect_loss_matches(result_a, a, "without preallocated output (miss)");
    expect_loss_matches(output_b, b, "into preallocated output");
    expect_loss_matches(output_c, c, "into a fresh preallocated output (hit)");
    expect_loss_matches(result_d, d, "without preallocated output (hit)");

    device.clear_program_cache();
    const auto e = make_inputs(N, H, W, seed + 8U);
    const auto output_e = make_output();
    const auto entries_before_preallocated_miss = device.num_program_cache_entries();
    const auto result_e = ttnn::prim::ttml_cross_entropy_fw(e.input, e.target, output_e);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_preallocated_miss)
        << "first preallocated launch did not populate the cache";

    const auto f = make_inputs(N, H, W, seed + 10U);
    const auto result_f = ttnn::prim::ttml_cross_entropy_fw(f.input, f.target, std::nullopt);

    const auto g = make_inputs(N, H, W, seed + 12U);
    const auto output_g = make_output();
    entries_before_hit = device.num_program_cache_entries();
    const auto result_g = ttnn::prim::ttml_cross_entropy_fw(g.input, g.target, output_g);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit)
        << "second preallocated launch was not a cache hit";

    EXPECT_EQ(result_e.buffer()->address(), output_e.buffer()->address());
    EXPECT_EQ(result_g.buffer()->address(), output_g.buffer()->address());

    expect_loss_matches(output_e, e, "into preallocated output, on an empty cache (miss)");
    expect_loss_matches(result_f, f, "without preallocated output, after a preallocated launch");
    expect_loss_matches(output_g, g, "into a fresh preallocated output, after a launch without one (hit)");
}

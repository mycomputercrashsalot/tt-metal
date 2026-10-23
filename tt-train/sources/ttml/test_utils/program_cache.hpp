// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tt-metalium/constants.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xoperation.hpp"
#include "xtensor/generators/xbuilder.hpp"

// Helpers for the program-cache tests, which launch an op again on fresh tensors while the earlier launches' tensors
// stay alive: a cache hit must read and write its own buffers on every core.

namespace ttml::test_utils {

// The device with its program cache enabled and empty, so that the calling test's first launch is a miss.
inline ttnn::distributed::MeshDevice& device_with_empty_program_cache() {
    auto& device = ttml::autograd::ctx().get_device();
    device.enable_program_cache();
    device.clear_program_cache();
    return device;
}

// The cores of the device's compute grid, which the program factories split their work over.
inline uint32_t num_worker_cores() {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

// Two work units per core and `extra` more. With 0 < extra < cores, split_work_to_cores gives `extra` cores three
// units and the others two: two core groups, so both of a factory's per-group kernels run, on any grid.
inline uint32_t two_core_group_units(uint32_t extra) {
    return 2U * num_worker_cores() + extra;
}

// The smallest count of `unit`-tile blocks whose tiles are more than twice the core count and not a multiple of it,
// so that their split has two core groups.
inline uint32_t blocks_for_two_core_groups(uint32_t unit) {
    const uint32_t cores = num_worker_cores();
    uint32_t blocks = 2U * cores / unit + 1U;
    if ((blocks * unit) % cores == 0U) {
        ++blocks;  // one step is enough: unit < cores, so the next total is not a multiple
    }
    return blocks;
}

// A sequence length S for an op that splits N * ceil(S / 32) tile rows over the cores. With N = 2 that is
// 2 * cores + 4 tile rows: four cores take three and the others two, so both core groups run on any grid. The last
// tile row of each batch entry holds 32 - `padding` rows.
inline uint32_t two_core_group_seq_len(uint32_t padding) {
    return (num_worker_cores() + 2U) * tt::constants::TILE_HEIGHT - padding;
}

// With N = 2, a sequence length that gives grid.y + 1 or grid.y + 2 tile rows, one per core: a factory that walks the
// grid column by column fills its first column and one or two cores of the second. A hit that walks the cores in
// another order misses some that got work, and one that walks further reaches idle cores, which have no runtime args.
// The last tile row of each batch entry holds 32 - `padding` rows.
inline uint32_t partial_grid_seq_len(uint32_t padding) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    return (static_cast<uint32_t>(grid.y) / 2U + 1U) * tt::constants::TILE_HEIGHT - padding;
}

// A bf16 TILE tensor of zeros, e.g. for a preallocated output.
inline ttnn::Tensor zeros_on_device(const std::vector<uint32_t>& shape) {
    const std::vector<std::size_t> xt_shape(shape.begin(), shape.end());
    return ttml::core::from_xtensor(xt::xarray<float>(xt::zeros<float>(xt_shape)), &ttml::autograd::ctx().get_device());
}

// A later launch's tensor must not share a buffer with an earlier one's, or a hit that kept the old address would pass.
inline void expect_new_buffer(const ttnn::Tensor& earlier, const ttnn::Tensor& later, const std::string& label) {
    EXPECT_NE(later.buffer()->address(), earlier.buffer()->address()) << label;
}

// A later launch must not write into an earlier launch's tensors: `tensor` still holds `snapshot`.
inline void expect_unchanged(const ttnn::Tensor& tensor, const xt::xarray<float>& snapshot, const std::string& label) {
    EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(tensor), snapshot)))
        << label << " changed after a later launch";
}

// Equal bits: tells -0 from 0, and a NaN equals the same NaN.
inline bool same_bits(float a, float b) {
    return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

// The elements whose bits differ; all of them if the shapes differ.
inline std::size_t count_bit_mismatches(const xt::xarray<float>& expected, const xt::xarray<float>& actual) {
    if (expected.shape() != actual.shape()) {
        return std::max(expected.size(), actual.size());
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        mismatches += static_cast<std::size_t>(!same_bits(expected.flat(i), actual.flat(i)));
    }
    return mismatches;
}

// A suite that opens the device once and starts every test from an empty program cache.
class ProgramCacheSuiteTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        device_with_empty_program_cache();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

}  // namespace ttml::test_utils

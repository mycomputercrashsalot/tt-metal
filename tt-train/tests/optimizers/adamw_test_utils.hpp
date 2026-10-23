// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <utility>

#include "core/tt_tensor_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "xtensor/containers/xarray.hpp"

// Launch tensors and checks shared by the AdamW and AdamWFullPrecision program-cache tests.

namespace ttml::test_utils::adamw {

// A device tensor and the values the kernels read from it.
struct StoredTensor {
    ttnn::Tensor tensor;
    xt::xarray<float> before;
};

inline StoredTensor store(ttnn::Tensor tensor) {
    auto before = ttml::core::to_xtensor(tensor);
    return {std::move(tensor), std::move(before)};
}

// The tensors of one AdamW launch.
struct AdamWLaunchTensors {
    StoredTensor param;
    StoredTensor grad;
    StoredTensor exp_avg;
    StoredTensor exp_avg_sq;
    std::optional<StoredTensor> max_exp_avg_sq;  // AMSGrad only
};

// Every tensor of a launch, as read back from the device or as a CPU reference computes it.
struct AdamWValues {
    xt::xarray<float> param;
    xt::xarray<float> grad;
    xt::xarray<float> exp_avg;
    xt::xarray<float> exp_avg_sq;
    std::optional<xt::xarray<float>> max_exp_avg_sq;  // AMSGrad only
};

inline AdamWValues read_back(const AdamWLaunchTensors& tensors) {
    AdamWValues values{
        ttml::core::to_xtensor(tensors.param.tensor),
        ttml::core::to_xtensor(tensors.grad.tensor),
        ttml::core::to_xtensor(tensors.exp_avg.tensor),
        ttml::core::to_xtensor(tensors.exp_avg_sq.tensor),
        std::nullopt};
    if (tensors.max_exp_avg_sq.has_value()) {
        values.max_exp_avg_sq = ttml::core::to_xtensor(tensors.max_exp_avg_sq->tensor);
    }
    return values;
}

inline void expect_bitwise_equal(const AdamWValues& expected, const AdamWValues& actual, const std::string& label) {
    EXPECT_EQ(count_bit_mismatches(expected.param, actual.param), 0U) << label << ": param";
    EXPECT_EQ(count_bit_mismatches(expected.grad, actual.grad), 0U) << label << ": grad";
    EXPECT_EQ(count_bit_mismatches(expected.exp_avg, actual.exp_avg), 0U) << label << ": exp_avg";
    EXPECT_EQ(count_bit_mismatches(expected.exp_avg_sq, actual.exp_avg_sq), 0U) << label << ": exp_avg_sq";
    ASSERT_EQ(expected.max_exp_avg_sq.has_value(), actual.max_exp_avg_sq.has_value()) << label;
    if (expected.max_exp_avg_sq.has_value()) {
        EXPECT_EQ(count_bit_mismatches(*expected.max_exp_avg_sq, *actual.max_exp_avg_sq), 0U)
            << label << ": max_exp_avg_sq";
    }
}

inline void expect_distinct_buffers(const AdamWLaunchTensors& a, const AdamWLaunchTensors& b) {
    EXPECT_NE(a.param.tensor.buffer()->address(), b.param.tensor.buffer()->address());
    EXPECT_NE(a.grad.tensor.buffer()->address(), b.grad.tensor.buffer()->address());
    EXPECT_NE(a.exp_avg.tensor.buffer()->address(), b.exp_avg.tensor.buffer()->address());
    EXPECT_NE(a.exp_avg_sq.tensor.buffer()->address(), b.exp_avg_sq.tensor.buffer()->address());
    if (a.max_exp_avg_sq.has_value() && b.max_exp_avg_sq.has_value()) {
        EXPECT_NE(a.max_exp_avg_sq->tensor.buffer()->address(), b.max_exp_avg_sq->tensor.buffer()->address());
    }
}

}  // namespace ttml::test_utils::adamw

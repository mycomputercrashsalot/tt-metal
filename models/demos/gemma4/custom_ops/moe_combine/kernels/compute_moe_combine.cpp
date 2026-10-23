// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for MoeCombine, accumulator-through-L1 form: per expert one acquire / release cycle. The running sum
// lives in the fp32 CB cb_acc between experts (copied back into dest 0, the new product multiplied into dest 1
// by the FPU column broadcast, added on the SFPU, packed back), so no FPU write ever races an SFPU read of the
// same dest tile; the last expert packs to the bf16 output.
// compile-time args: [cb_d, cb_w, cb_out, E, cb_acc]; runtime args: [column count]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_D = get_compile_time_arg_val(0);
constexpr uint32_t CB_W = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
constexpr uint32_t E = get_compile_time_arg_val(3);
constexpr uint32_t CB_ACC = get_compile_time_arg_val(4);

void kernel_main() {
    compute_kernel_hw_startup(CB_D, CB_W, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer cbd(CB_D), cbw(CB_W), cbo(CB_OUT), cba(CB_ACC);
    for (uint32_t c = 0; c < count; ++c) {
        for (uint32_t e = 0; e < E; ++e) {
            const bool last = (e + 1 == E);
            cbd.wait_front(1);
            cbw.wait_front(1);
            tile_regs_acquire();
            if (e > 0) {
                cba.wait_front(1);
                reconfig_data_format_srca(CB_ACC);
                copy_init(CB_ACC);
                copy_tile(CB_ACC, 0, 0);
                cba.pop_front(1);
            }
            reconfig_data_format(CB_D, CB_W);
            mul_bcast_cols_init(CB_D, CB_W);
            mul_tiles_bcast_cols(CB_D, CB_W, 0, 0, e > 0 ? 1 : 0);
            if (e > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (last) {
                pack_reconfig_data_format(CB_OUT);
                cbo.reserve_back(1);
                pack_tile(0, CB_OUT, 0);
                cbo.push_back(1);
            } else {
                pack_reconfig_data_format(CB_ACC);
                cba.reserve_back(1);
                pack_tile(0, CB_ACC, 0);
                cba.push_back(1);
            }
            tile_regs_release();
            cbd.pop_front(1);
            cbw.pop_front(1);
        }
    }
}

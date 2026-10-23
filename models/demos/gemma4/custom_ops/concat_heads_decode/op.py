# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``ConcatHeadsDecode``: the decode SDPA output ``[1, B, H(pad 32), D]`` (bf16, TILE, DRAM interleaved: one row
tile per user, the heads on its rows) to the o_proj input ``[1, 1, B, H*D]`` (bf16, TILE, DRAM interleaved: user
``b`` on row ``b``, head ``h`` at columns ``h*D..``) as one program: output tile ``(h, c)`` row ``b`` = input tile
``(b, c)`` row ``h``; rows ``>= B`` are zero. A core owns one column tile ``c`` of the head dim: it reads the ``B``
input tiles ``(b, c)`` once and assembles the ``H`` output tiles with 32-byte face-row copies. This replaces
``to_memory_config (interleaved -> height sharded), nlp_concat_heads_decode, sharded_to_interleaved`` (and the
``[:, :, :B]`` slice at B < 32): 3-4 programs -> 1.
"""

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class ConcatHeadsDecode:
    CB_IN = 0  # the B input tiles of this core's column
    CB_OUT = 1  # the H output tiles

    @staticmethod
    def golden(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """x [1, B, H, D] -> [1, 1, B, H*D]."""
        B, H, D = x.shape[1], num_heads, x.shape[3]
        return x[:, :, :H, :].reshape(1, 1, B, H * D)

    @classmethod
    def op(cls, x: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        _, B, Hp, D = (int(v) for v in x.shape)
        assert B <= TILE and x.padded_shape[2] == TILE and num_heads <= Hp, (x.shape, num_heads)
        assert D % TILE == 0
        cols = D // TILE
        device = x.device()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B, num_heads * D]), x.dtype, x.layout, device, x.memory_config()
        )
        grid = device.compute_with_storage_grid_size()
        n_cores = min(cols, grid.x * grid.y)
        per = [(cols // n_cores) + (1 if i < cols % n_cores else 0) for i in range(n_cores)]
        starts = [sum(per[:i]) for i in range(n_cores)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        tile_bytes = TILE * TILE * 2

        def cb(idx, pages):
            return ttnn.CBDescriptor(
                total_size=pages * tile_bytes,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
                ],
            )

        cbs = [cb(cls.CB_IN, TILE), cb(cls.CB_OUT, num_heads)]
        ct = [cls.CB_IN, cls.CB_OUT, B, num_heads, cols]
        ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            rt[core.x][core.y] = [x.buffer_address(), out.buffer_address(), starts[i], per[i]]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "concat_heads_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, out], program)

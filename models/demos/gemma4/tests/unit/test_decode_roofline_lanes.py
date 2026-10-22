# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-step op profile of the Galaxy lane engine (8x4 mesh: fractured dense weights, four lanes of 32
slots, routed experts on the decode MoE kernel) for the roofline split.

Same signpost / profiler-flush protocol as ``test_decode_roofline.py`` (the companion
``decode_roofline_report.py`` reads the signposted steps and the ops json), parametrized by the number
of active users per lane so the one-user step and the full 32-users-per-lane step can be compared op
by op. Inactive slots stay in the lane-major batch with position -1, exactly as the served engine pads
them, so ``upl_1`` is the serving batch-1 step and ``upl_32`` the 128-user step.

    export TT_METAL_DEVICE_PROFILER=1
    HF_MODEL=google/gemma-4-26B-A4B-it GEMMA4_KV_BFP8=1 GEMMA4_CCL_TOPOLOGY=linear \\
    GEMMA4_HYBRID_KV_CACHE_GROUPS=1 GEMMA4_PAGE_BLOCK_SIZE=64 \\
    python -m tracy -p -r -v -m pytest \\
        models/demos/gemma4/tests/unit/test_decode_roofline_lanes.py -k upl_32 -s
    python tools/tracy/process_ops_logs.py --date
    python models/demos/gemma4/tests/unit/decode_roofline_report.py <ops_perf_results.csv> \\
        --ops-json generated/profiler/roofline_ops_lanes_32.json --steps 3 --params-b 26 --chips 8
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tests.unit.test_decode_roofline import (
    N_STEPS,
    PREFILL_LEN,
    WARM_STEPS,
    _device_ops_in_order,
    _signpost,
)

SLOTS_PER_LANE = 32
BLOCKS_PER_USER = 4  # 256 tokens at block 64: the 128-token prompt plus the warm and profiled steps


@parametrize_mesh_with_fabric(
    mesh_shapes=[(8, 4)],
    # the decode MoE kernel needs the ring fabric; the served entry keeps the collective semaphores in L1_SMALL
    device_params_extra={"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "l1_small_size": 65536},
)
@pytest.mark.parametrize("users_per_lane", [1, 32], ids=["upl_1", "upl_32"])
def test_decode_roofline_lanes(mesh_device, users_per_lane, monkeypatch):
    skip_if_config_only_checkpoint()
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "1"))

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it")
    # Per-column pool: block 0 is the lane scratch, then 32 slots per lane (see test_lanes_e2e.py).
    pool_blocks = 1 + SLOTS_PER_LANE * BLOCKS_PER_USER
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    lanes = mesh_cfg.lanes
    B_g = lanes * SLOTS_PER_LANE
    rows = [lane * SLOTS_PER_LANE + s for lane in range(lanes) for s in range(users_per_lane)]
    page_rows = {
        row: torch.arange(
            1 + (row % SLOTS_PER_LANE) * BLOCKS_PER_USER,
            1 + (row % SLOTS_PER_LANE + 1) * BLOCKS_PER_USER,
            dtype=torch.int32,
        )
        for row in rows
    }

    prompt = tokenizer.encode("It was the best of times, it was the worst of times, " * 12, add_special_tokens=True)
    tok1 = torch.tensor(prompt[:PREFILL_LEN], dtype=torch.long).unsqueeze(0)
    cur_tok, positions = {}, {}
    for row in rows:
        out = generator.prefill_forward_text(
            tok1,
            page_table=page_rows[row].unsqueeze(0),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([PREFILL_LEN]),
            empty_slots=[row],
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        cur_tok[row] = int(torch.argmax(lg.reshape(-1)).item())
        positions[row] = PREFILL_LEN
    logger.info(f"prefilled {len(rows)} users ({users_per_lane} per lane x {lanes} lanes) of the {B_g}-row batch")

    def step():
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, BLOCKS_PER_USER, dtype=torch.int32)
        for row in rows:
            tokens_g[row] = cur_tok[row]
            pos_g[row] = positions[row]
            pt_g[row] = page_rows[row]
        logits = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=False,
            read_from_device=True,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        lg = logits[0] if isinstance(logits, (list, tuple)) else logits
        lg = lg.reshape(B_g, -1)
        for row in rows:
            cur_tok[row] = int(torch.argmax(lg[row]).item())
            positions[row] += 1

    for _ in range(WARM_STEPS):
        step()
    ttnn.synchronize_device(mesh_device)

    # Op names in execution order for ONE step (see test_decode_roofline.py: the device rows carry no names).
    ops_json = os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", f"roofline_ops_lanes_{users_per_lane}.json"
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    step()
    ttnn.synchronize_device(mesh_device)
    captured = ttnn.graph.end_graph_capture()
    ops = _device_ops_in_order(captured)
    os.makedirs(os.path.dirname(ops_json), exist_ok=True)
    with open(ops_json, "w") as f:
        json.dump({"batch_size": B_g, "users_per_lane": users_per_lane, "ops": ops}, f)
    logger.info(f"captured {len(ops)} device ops for one lane decode step -> {ops_json}")

    def _flush():
        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass

    _flush()
    # Bound the device log: everything so far (load, 128 prefills, warm steps) is flushed and not needed; keep
    # only the two header lines so the profiled steps' rows are the file (the 8x4 mesh writes ~33 GB otherwise).
    dev_log = os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv"
    )
    if os.path.exists(dev_log):
        with open(dev_log) as f:
            header = [next(f), next(f)]
        with open(dev_log, "w") as f:
            f.writelines(header)
        logger.info(f"device log truncated to its header before the profiled steps: {dev_log}")
    _signpost("start")
    for _ in range(N_STEPS):
        step()
        ttnn.synchronize_device(mesh_device)
        _flush()
    _signpost("stop")
    _flush()
    logger.info(
        f"{os.path.basename(model_path)} lanes={lanes} users_per_lane={users_per_lane}: {N_STEPS} eager decode steps between signposts"
    )

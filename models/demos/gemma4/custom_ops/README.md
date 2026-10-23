# Gemma 4 custom ops

Model-local fused kernels (`ttnn.generic_op`), the GLM 5.3 `custom_ops/` pattern. The lane decode step of the
26B-A4B on the Blackhole Galaxy runs ~2400 device programs per step (122 top-level ttnn ops per layer) and
54 ms of device time of which ~27 ms is data movement (reshapes, layout and memory-config conversions, slices,
repeats, pads), against a 1.7 ms DRAM floor. Each op here removes one such chain; the measurement protocol is
`tests/unit/test_decode_roofline_lanes.py` (device ops/step, device ms/step) plus the lanes e2e for answers.

| op | replaces (per layer, decode) | status |
|---|---|---|
| `rope_decode` | q/k RoPE as `(getitem x2, neg, concat, multiply, add) x2, multiply x2` per rotation (~8 programs after the per-step cos/sin hoist) | done: 9 unit tests (pcc 0.99999x vs golden and composite, identity exact), 64.7 -> 7.2 us device per b32h16d256 call (13 -> 1 programs); opt-in `GEMMA4_ROPE_DECODE_CUSTOM_OP=1` in `attention/decode.py` |
| `geglu_routed_expert` | the shared `unified_routed_expert_ffn` edited for GeGLU (dispatch-prefill experts): now the only prefill expert path, TtRoutedExpert is the weight loader | done: bit-identical to the shared op with GeGLU, e2e dispatch PCC 0.99925 vs HF (128 / 1024 / 4224 tokens), same device time (541 us per 5633 rows per chip) |
| `norm_reshard` (planned) | `to_memory_config -> rms_norm -> sharded_to_interleaved` triples (11/layer) | keep the residual stream sharded first |
| `router_topk` (planned) | `softmax, topk, sum, div` + the kernel-side re-top-k | sparse routing pass-through landed first |

Layout of an op folder:

```
custom_ops/<op>/op.py        # class <Op>: golden(torch) + op(ttnn tensors) -> ttnn.generic_op(...)
custom_ops/<op>/kernels/     # reader_<op>.cpp, compute_<op>.cpp, writer_<op>.cpp
tests/unit/custom_ops/test_<op>.py
```

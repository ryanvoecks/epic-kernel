---
name: project_mixtral_status
description: Current correctness status of Mixtral megakernel — all SMALL_TEST stages passing as of 2026-03-24
type: project
---

All 8 opcode stages of the Mixtral megakernel SMALL_TEST now PASS via `uv run python debug_mixtral_mk.py --no-stop` run from `demos/mixtral/`.

**Why:** Three scalar bypass fixes were applied to eliminate WGMMA swizzle corruption, plus a `MIXTRAL_NUM_LAYERS` alignment fix for the lm_head barrier index. See `FAILURE.md` for full details.

**How to apply:** The SMALL_TEST correctness baseline is established. Next work should target full-scale (non-SMALL_TEST) correctness testing with actual Mixtral 8x7B weights, or performance optimization.

---
name: feedback_build_command
description: Build command must use top-level make from repo root, not subdirectory make
type: feedback
---

Always build with `uv run make mk_mixtral` from the repo root `/home/of222/epic-kernel`, not `uv run make -C demos/mixtral mk_mixtral_small`. The top-level Makefile exports `BUILD_DIR`, `PYTHON`, `THUNDERKITTENS_ROOT`, and `MEGAKERNELS_ROOT` that the subdirectory Makefile depends on.

**Why:** Running make from the subdirectory fails with "No rule to make target" or missing env vars because those exports only happen at the top level.

**How to apply:** Any time a build is needed, use `uv run make mk_mixtral` from `/home/of222/epic-kernel`.

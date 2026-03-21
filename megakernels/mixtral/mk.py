"""Stub MK interpreter for Mixtral — CUDA kernel not yet implemented."""
from pathlib import Path

from megakernels.mk import MK_Interpreter
from megakernels.mixtral.instructions import MixtralGlobals


class MixtralMK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        raise NotImplementedError(
            "Mixtral CUDA megakernel not yet built. "
            "Run with mode=pyvm for the Python reference implementation."
        )

    def interpret(self, globs: MixtralGlobals):
        raise NotImplementedError

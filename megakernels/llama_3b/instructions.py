# Re-export all instruction types from the latency module.
# The 3B model uses the same instruction opcodes and Globals layout;
# only block sizes and dimension values differ at runtime.
from megakernels.demos.latency.instructions import (  # noqa: F401
    AttentionReduction,
    DownProjResidual,
    Globals,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    RMS_LM_Head,
)

from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from megakernels.llama import (
    RMSNorm,
    apply_rotary_pos_emb_interleaved,
    attention,
)
from megakernels.model_types import BatchState, DeviceType, ExtraModelConfig
from megakernels.utils import load_safetensors_repo

KV_Cache = tuple[Tensor, Tensor]


class MixtralAttention(nn.Module):
    def __init__(self, config, extra_config: ExtraModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size or 1
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.input_layernorm = _RMSNorm(config)

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.kv_cache: KV_Cache | None = None

    def forward(self, batch_state: BatchState) -> BatchState:
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert batch_state.position_ids is not None
        assert self.kv_cache is not None
        assert batch_state.seq_len is not None

        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)

        bsz, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, -1)

        cos, sin = batch_state.position_embeddings
        dtype = query_states.dtype

        if self.extra_config.interleave_rope:
            rope_fn = apply_rotary_pos_emb_interleaved
        else:
            rope_fn = apply_rotary_pos_emb

        query_states, key_states = rope_fn(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=-2,
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )

        attn_output = raw_attn_output.reshape(bsz, seq_len, -1)
        o_proj = self.o_proj(attn_output)

        batch_state.hidden_states = residual + o_proj
        return batch_state


class _RMSNorm(nn.Module):
    """Standalone RMSNorm that takes a config with hidden_size and rms_norm_eps."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)
        return self.weight * hidden_states.to(input_dtype)


class MixtralExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MixtralSparseMoE(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_attention_layernorm = _RMSNorm(config)
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            [MixtralExpert(config) for _ in range(config.num_local_experts)]
        )

    def forward(self, batch_state: BatchState) -> BatchState:
        inp = batch_state.hidden_states
        assert inp is not None

        hidden_states = self.post_attention_layernorm(inp)

        # Router: [hidden] -> [num_experts]
        router_logits = self.gate(hidden_states)  # [bsz, seq, num_experts]
        scores = F.softmax(router_logits.float(), dim=-1)

        # Top-k selection
        top_weights, top_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # Renormalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights.to(hidden_states.dtype)

        # Mix expert outputs
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        final_hidden = torch.zeros_like(hidden_states_flat)

        top_indices_flat = top_indices.view(-1, self.num_experts_per_tok)
        top_weights_flat = top_weights.view(-1, self.num_experts_per_tok)

        # For each expert, gather tokens that selected it
        for expert_idx, expert in enumerate(self.experts):
            mask = (top_indices_flat == expert_idx).any(dim=-1)  # [bsz*seq]
            if not mask.any():
                continue
            expert_input = hidden_states_flat[mask]
            expert_out = expert(expert_input)

            # Get the weight for this expert for each selected token
            slot = (top_indices_flat[mask] == expert_idx).float()
            weights = (top_weights_flat[mask] * slot).sum(dim=-1, keepdim=True)

            final_hidden[mask] += weights * expert_out

        batch_state.hidden_states = inp + final_hidden.view(bsz, seq_len, hidden_dim)
        return batch_state


class MixtralBlock(nn.Module):
    def __init__(self, config, extra_config: ExtraModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MixtralAttention(config, extra_config, layer_idx)
        self.moe = MixtralSparseMoE(config, layer_idx)

    def forward(self, batch_state: BatchState) -> BatchState:
        batch_state = self.self_attn(batch_state)
        batch_state = self.moe(batch_state)
        return batch_state


class MixtralLMHead(nn.Module):
    def __init__(self, config, extra_config: ExtraModelConfig):
        super().__init__()
        self.input_norm = _RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, batch_state: BatchState) -> BatchState:
        assert batch_state.hidden_states is not None
        hidden_states = self.input_norm(batch_state.hidden_states)
        logits = self.lm_head(hidden_states)
        batch_state.output_ids = logits.argmax(dim=-1)
        return batch_state


class MixtralEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState) -> BatchState:
        batch_state.hidden_states = self.embed_tokens(batch_state.input_ids)
        return batch_state


class MixtralModel(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(self, config, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = MixtralEmbeddings(config)

        self.layers = nn.ModuleList(
            [MixtralBlock(config, extra_config, i) for i in range(config.num_hidden_layers)]
        )

        # Use Llama RoPE (same formula)
        self.rope = LlamaRotaryEmbedding(config=config)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy = torch.empty((0, config.hidden_size), dtype=torch.float32)
        cos, sin = self.rope(dummy, position_ids)
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def interleave_rope(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        half_head_dim = head_dim // 2
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        indices_for_q_list = []
        for n in range(num_q_heads):
            offset = n * head_dim
            for i in range(half_head_dim):
                indices_for_q_list.append(i + offset)
                indices_for_q_list.append(i + half_head_dim + offset)

        indices_for_q = torch.tensor(indices_for_q_list, device=self.rope_cos.device)
        one_head_indices = indices_for_q[:head_dim]

        self.rope_cos = self.rope_cos[..., one_head_indices]
        self.rope_sin = self.rope_sin[..., one_head_indices]

        indices_for_k = indices_for_q[:head_dim * num_kv_heads]

        for mod in self.modules():
            if isinstance(mod, MixtralAttention):
                mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
                mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]

    def forward(self, batch_state: BatchState) -> BatchState:
        out = self.embed_tokens(batch_state)
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)
        for layer in self.layers:
            out = layer(out)
        return out


@dataclass
class MixtralStackedParams:
    qkv_proj: Tensor        # [num_layers, (num_q + 2*num_kv) * head_dim, hidden]
    o_proj: Tensor          # [num_layers, hidden, num_q * head_dim]
    attn_ln_weight: Tensor  # [num_layers, hidden]
    ffn_ln_weight: Tensor   # [num_layers, hidden]
    router_weight: Tensor   # [num_layers, num_experts, hidden]
    expert_gate: Tensor     # [num_layers, num_experts, intermediate, hidden]
    expert_up: Tensor       # [num_layers, num_experts, intermediate, hidden]
    expert_down: Tensor     # [num_layers, num_experts, hidden, intermediate]


class MixtralForCausalLM(nn.Module):
    def __init__(self, config, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = MixtralModel(config, extra_config)
        self.lm_head = MixtralLMHead(config, extra_config)

    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads // self.extra_config.tp_size

    def num_qo_heads(self) -> int:
        return self.config.num_attention_heads // self.extra_config.tp_size

    def head_dim(self) -> int:
        return self.config.hidden_size // self.config.num_attention_heads

    def forward(self, batch_state: BatchState) -> BatchState:
        input_ids = batch_state.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        position_ids = batch_state.position_ids
        if position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        out = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=batch_state.hidden_states,
            seq_len=batch_state.seq_len,
        )
        out = self.model(out)
        out = self.lm_head(out)
        return out

    def setup_caches(self):
        config = self.config
        head_dim = config.hidden_size // config.num_attention_heads
        max_len = (
            self.extra_config.max_len_override or config.max_position_embeddings
        )
        k_cache = torch.zeros(
            (config.num_hidden_layers, self.extra_config.max_batch_size, max_len,
             config.num_key_value_heads, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()
        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(config.num_hidden_layers):
            block: MixtralBlock = self.model.layers[layer_idx]  # type: ignore
            block.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MixtralForCausalLM":
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config = AutoConfig.from_pretrained(model_name_or_path)

        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            dtype = getattr(config, "torch_dtype", torch.bfloat16)

        with init_empty_weights(include_buffers=False):
            model = cls(config, extra_config)
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )
            model_path = Path(snapshot)

        model.load_from_safetensors(model_path)
        model.to(device=device)
        model.requires_grad_(False)

        if extra_config.interleave_rope:
            model.model.interleave_rope()

        model.stack_params()
        model.setup_caches()
        return model

    def _make_name_to_hf_name(self) -> dict[str, str]:
        """Map our parameter names to HuggingFace weight names."""
        config = self.config
        name_to_hf: dict[str, str] = {}

        for i in range(config.num_hidden_layers):
            # Attention layernorm
            name_to_hf[f"model.layers.{i}.self_attn.input_layernorm.weight"] = (
                f"model.layers.{i}.input_layernorm.weight"
            )
            # QKV
            name_to_hf[f"model.layers.{i}.self_attn.q_proj.weight"] = (
                f"model.layers.{i}.self_attn.q_proj.weight"
            )
            name_to_hf[f"model.layers.{i}.self_attn.k_proj.weight"] = (
                f"model.layers.{i}.self_attn.k_proj.weight"
            )
            name_to_hf[f"model.layers.{i}.self_attn.v_proj.weight"] = (
                f"model.layers.{i}.self_attn.v_proj.weight"
            )
            name_to_hf[f"model.layers.{i}.self_attn.o_proj.weight"] = (
                f"model.layers.{i}.self_attn.o_proj.weight"
            )
            # MoE layernorm
            name_to_hf[f"model.layers.{i}.moe.post_attention_layernorm.weight"] = (
                f"model.layers.{i}.post_attention_layernorm.weight"
            )
            # Router
            name_to_hf[f"model.layers.{i}.moe.gate.weight"] = (
                f"model.layers.{i}.block_sparse_moe.gate.weight"
            )
            # Expert weights
            for j in range(config.num_local_experts):
                name_to_hf[f"model.layers.{i}.moe.experts.{j}.w1.weight"] = (
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"
                )
                name_to_hf[f"model.layers.{i}.moe.experts.{j}.w2.weight"] = (
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"
                )
                name_to_hf[f"model.layers.{i}.moe.experts.{j}.w3.weight"] = (
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"
                )

        # Embeddings
        name_to_hf["model.embed_tokens.embed_tokens.weight"] = "model.embed_tokens.weight"
        # Final norm & LM head
        name_to_hf["lm_head.input_norm.weight"] = "model.norm.weight"
        name_to_hf["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf

    def load_from_safetensors(self, model_path: Path):
        name_to_hf = self._make_name_to_hf_name()
        all_hf_names = set(name_to_hf.values())

        hf_state = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map={},
        )

        state_dict = {k: hf_state[v] for k, v in name_to_hf.items()}
        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        config = self.config
        layers: list[MixtralBlock] = self.model.layers  # type: ignore

        def stack(modules, attr):
            params = [getattr(m, attr) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, attr)[:] = stacked[i]
            return stacked

        self_attns = [b.self_attn for b in layers]
        moes = [b.moe for b in layers]

        # O proj and attn LN
        stacked_o_proj = stack([a.o_proj for a in self_attns], "weight")
        stacked_attn_ln = stack([a.input_layernorm for a in self_attns], "weight")

        # FFN LN (pre-MoE)
        stacked_ffn_ln = stack([m.post_attention_layernorm for m in moes], "weight")

        # Router weights: [num_layers, num_experts, hidden]
        stacked_router = torch.stack([m.gate.weight for m in moes], dim=0)
        for i, m in enumerate(moes):
            m.gate.weight[:] = stacked_router[i]

        # QKV: concatenate q, k, v per layer then stack
        qkv_weights = []
        for attn in self_attns:
            cat = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
            qkv_weights.append(cat)
        stacked_qkv = torch.stack(qkv_weights, dim=0)

        # Re-assign split q, k, v
        q_dim = config.num_attention_heads * self.head_dim()
        k_dim = config.num_key_value_heads * self.head_dim()
        for i, attn in enumerate(self_attns):
            q, k, v = stacked_qkv[i].split([q_dim, k_dim, k_dim], dim=0)
            attn.q_proj.weight[:] = q
            attn.k_proj.weight[:] = k
            attn.v_proj.weight[:] = v

        # Expert weights: [num_layers, num_experts, ...]
        num_experts = config.num_local_experts
        expert_gate_list = []
        expert_up_list = []
        expert_down_list = []
        for m in moes:
            gate_layer = torch.stack([m.experts[j].w1.weight for j in range(num_experts)], dim=0)
            up_layer = torch.stack([m.experts[j].w3.weight for j in range(num_experts)], dim=0)
            down_layer = torch.stack([m.experts[j].w2.weight for j in range(num_experts)], dim=0)
            expert_gate_list.append(gate_layer)
            expert_up_list.append(up_layer)
            expert_down_list.append(down_layer)

        stacked_expert_gate = torch.stack(expert_gate_list, dim=0)
        stacked_expert_up = torch.stack(expert_up_list, dim=0)
        stacked_expert_down = torch.stack(expert_down_list, dim=0)

        # Assign back
        for i, m in enumerate(moes):
            for j in range(num_experts):
                m.experts[j].w1.weight[:] = stacked_expert_gate[i, j]
                m.experts[j].w3.weight[:] = stacked_expert_up[i, j]
                m.experts[j].w2.weight[:] = stacked_expert_down[i, j]

        self.stacked_params = MixtralStackedParams(
            qkv_proj=stacked_qkv,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_attn_ln,
            ffn_ln_weight=stacked_ffn_ln,
            router_weight=stacked_router,
            expert_gate=stacked_expert_gate,
            expert_up=stacked_expert_up,
            expert_down=stacked_expert_down,
        )

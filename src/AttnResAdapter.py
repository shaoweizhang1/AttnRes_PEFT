import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.generation import GenerationMixin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class DepthAttentionAdapter(nn.Module):
    """
    PEFT-safe AttnRes adapter.

    It keeps the frozen backbone residual path intact and injects a gated depth
    routed correction. Optional upgrades are backward-compatible:
      - low-rank input-dependent query
      - token-conditioned gate multiplier
    """
    def __init__(
        self,
        hidden_size,
        gate_init=0.0,
        query_rank=0,
        token_gate=False,
        residual_mode="add",
        projection_rank=0,
        projection_init_std=0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_rank = query_rank
        self.token_gate = token_gate
        self.residual_mode = residual_mode
        self.projection_rank = projection_rank
        self.projection_init_std = projection_init_std

        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self.score_norm = RMSNorm(hidden_size)
        self.out_norm = RMSNorm(hidden_size)
        self.query_norm = RMSNorm(hidden_size)

        if query_rank and query_rank > 0:
            self.query_down = nn.Linear(hidden_size, query_rank, bias=False)
            self.query_up = nn.Linear(query_rank, hidden_size, bias=False)
            nn.init.zeros_(self.query_up.weight)
        else:
            self.query_down = None
            self.query_up = None

        if token_gate:
            self.token_gate_proj = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.token_gate_proj.weight)
            nn.init.zeros_(self.token_gate_proj.bias)
        else:
            self.token_gate_proj = None

        if residual_mode not in {"add", "mul"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        if residual_mode == "mul":
            rank = projection_rank if projection_rank and projection_rank > 0 else max(1, hidden_size // 16)
            self.mod_norm = RMSNorm(hidden_size)
            self.mod_down = nn.Linear(hidden_size, rank, bias=False)
            self.mod_up = nn.Linear(rank, hidden_size, bias=False)
            nn.init.normal_(self.mod_up.weight, mean=0.0, std=projection_init_std)
        else:
            self.mod_norm = None
            self.mod_down = None
            self.mod_up = None

    def forward(self, h_base, prev_states):
        """
        h_base: [B, T, D], output from the frozen model layer.
        prev_states: list of [B, T, D], depth memory states.
        """
        output_dtype = h_base.dtype
        compute_dtype = self.query.dtype

        sources = torch.stack(prev_states, dim=1).to(compute_dtype)
        h_base_compute = h_base.to(compute_dtype)

        normed = self.score_norm(sources)
        if self.query_down is None:
            scores = torch.einsum("d,bltd->blt", self.query, normed)
        else:
            query = self.query.view(1, 1, self.hidden_size)
            query = query + self.query_up(self.query_down(self.query_norm(h_base_compute)))
            scores = torch.einsum("btd,bltd->blt", query, normed)

        alpha = F.softmax(scores, dim=1)
        delta = torch.einsum("blt,bltd->btd", alpha, sources)
        delta = self.out_norm(delta)

        gate = self.gate
        if self.token_gate_proj is not None:
            gate = gate * torch.sigmoid(self.token_gate_proj(h_base_compute))

        if self.residual_mode == "mul":
            scale = self.mod_up(F.silu(self.mod_down(self.mod_norm(delta))))
            out = h_base_compute * (1.0 + gate * torch.tanh(scale))
        else:
            out = h_base_compute + gate * delta
        return out.to(output_dtype), alpha


class Qwen3AttnResConfigMixin:
    attnres_lookback = None
    attnres_enable = True


class Qwen3ForCausalLMWithAttnRes(PreTrainedModel, GenerationMixin):
    """
    Wrapper around a Qwen causal LM with frozen backbone and trainable
    depth-attention residual adapters.
    """
    config_class = None
    base_model_prefix = "model"

    def __init__(
        self,
        config,
        base_model,
        lookback=None,
        gate_init=0.0,
        block_size=None,
        block_reduce="mean",
        query_rank=0,
        token_gate=False,
        residual_mode="add",
        projection_rank=0,
        projection_init_std=0.02,
        memory_mode="hybrid",
    ):
        if getattr(config, "_attn_implementation", None) != "eager":
            config._attn_implementation = "eager"
            config._attn_implementation_internal = "eager"
        super().__init__(config)
        self.model = base_model.model
        self.lm_head = base_model.lm_head
        self.base_lm = base_model
        self.lookback = lookback
        self.gate_init = gate_init
        self.block_size = block_size
        self.block_reduce = block_reduce
        self.query_rank = query_rank
        self.token_gate = token_gate
        self.residual_mode = residual_mode
        self.projection_rank = projection_rank
        self.projection_init_std = projection_init_std
        self.memory_mode = memory_mode

        hidden_size = self.model.embed_tokens.embedding_dim
        self.adapters = nn.ModuleList([
            DepthAttentionAdapter(
                hidden_size=hidden_size,
                gate_init=self.gate_init,
                query_rank=self.query_rank,
                token_gate=self.token_gate,
                residual_mode=self.residual_mode,
                projection_rank=self.projection_rank,
                projection_init_std=self.projection_init_std,
            )
            for _ in range(len(self.model.layers))
        ])

        emb = self.model.embed_tokens.weight
        # Keep trainable adapter params in fp32 so AMP/fp16 training can unscale gradients safely.
        self.adapters.to(device=emb.device, dtype=torch.float32)
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.base_lm.parameters():
            p.requires_grad = False
        for p in self.adapters.parameters():
            p.requires_grad = True

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}"
        )

    def adapter_state_dict(self):
        return self.adapters.state_dict()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        adapter_state = self.adapter_state_dict()
        if destination is None:
            destination = {}
        for name, tensor in adapter_state.items():
            key = f"{prefix}adapters.{name}"
            destination[key] = tensor if keep_vars else tensor.detach().clone()
        return destination

    def save_pretrained(self, save_directory, *args, **kwargs):
        safe_serialization = kwargs.pop("safe_serialization", True)
        selected_state_dict = kwargs.pop("state_dict", None)
        if selected_state_dict is None:
            selected_state_dict = self.state_dict()
        return super().save_pretrained(
            save_directory,
            *args,
            state_dict=selected_state_dict,
            safe_serialization=safe_serialization,
            **kwargs,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_output_embeddings):
        self.lm_head = new_output_embeddings

    def _adapters_effectively_disabled(self):
        return all(torch.allclose(adapter.gate.detach(), torch.zeros_like(adapter.gate)) for adapter in self.adapters)

    def _compress_blocks(self, states):
        if self.block_size is None or self.block_size <= 0 or not states:
            return states

        block_states = []
        for start in range(0, len(states), self.block_size):
            block = states[start:start + self.block_size]
            if block:
                stacked = torch.stack(block, dim=0)
                if self.block_reduce == "mean":
                    block_states.append(stacked.mean(dim=0))
                elif self.block_reduce == "sum":
                    block_states.append(stacked.sum(dim=0))
                else:
                    raise ValueError(f"Unsupported block_reduce: {self.block_reduce}")
        return block_states

    def _build_depth_memory(self, prev_states):
        if not prev_states:
            return prev_states

        embedding_state = prev_states[:1]
        layer_states = prev_states[1:]

        if self.memory_mode == "lookback":
            selected = layer_states if self.lookback is None else layer_states[-self.lookback:]
            memory = embedding_state + selected
        elif self.memory_mode == "block":
            memory = embedding_state + self._compress_blocks(layer_states)
        elif self.memory_mode == "hybrid":
            if self.lookback is None:
                recent_states = layer_states
                older_states = []
            else:
                recent_states = layer_states[-self.lookback:]
                older_states = layer_states[:-self.lookback]
            memory = embedding_state + self._compress_blocks(older_states) + recent_states
        else:
            raise ValueError(f"Unsupported memory_mode: {self.memory_mode}")

        return memory if memory else prev_states

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs.pop("cache_position", None)
        kwargs.pop("position_embeddings", None)
        kwargs.pop("use_cache", None)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            **kwargs,
        }

    def generate(self, *args, **kwargs):
        force_attnres_path = kwargs.pop("force_attnres_path", True)
        strict_base_when_disabled = kwargs.pop("strict_base_when_disabled", False)
        if self._adapters_effectively_disabled() and not force_attnres_path:
            return self.base_lm.generate(*args, **kwargs)
        kwargs["force_attnres_path"] = force_attnres_path
        kwargs["strict_base_when_disabled"] = strict_base_when_disabled
        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        force_attnres_path=True,
        strict_base_when_disabled=False,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        kwargs.pop("do_sample", None)

        if self._adapters_effectively_disabled() and not force_attnres_path:
            return self.base_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        prev_states = []
        hook_state = {"initialized": False}
        handles = []

        def make_layer_hook(adapter):
            def layer_hook(_module, layer_inputs, layer_output):
                h_in = layer_inputs[0]
                if not hook_state["initialized"]:
                    prev_states.append(h_in)
                    hook_state["initialized"] = True

                h_base = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                adapter_prev_states = self._build_depth_memory(prev_states)
                h_new, _alpha = adapter(h_base, adapter_prev_states)
                prev_states.append(h_new)

                if isinstance(layer_output, tuple):
                    return (h_new, *layer_output[1:])
                return h_new

            return layer_hook

        for layer, adapter in zip(self.model.layers, self.adapters):
            handles.append(layer.register_forward_hook(make_layer_hook(adapter)))

        try:
            return self.base_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()


def load_qwen3_attnres_model(
    model_name_or_path,
    lookback=8,
    gate_init=0.0,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    block_size=None,
    block_reduce="mean",
    query_rank=0,
    token_gate=False,
    residual_mode="add",
    projection_rank=0,
    projection_init_std=0.02,
    memory_mode="hybrid",
):
    """Load or wrap a causal LM with the AttnRes adapter."""
    if isinstance(model_name_or_path, str):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
    else:
        base_model = model_name_or_path

    base_model.config._attn_implementation = "eager"
    base_model.config._attn_implementation_internal = "eager"

    wrapped = Qwen3ForCausalLMWithAttnRes(
        config=base_model.config,
        base_model=base_model,
        lookback=lookback,
        gate_init=gate_init,
        block_size=block_size,
        block_reduce=block_reduce,
        query_rank=query_rank,
        token_gate=token_gate,
        residual_mode=residual_mode,
        projection_rank=projection_rank,
        projection_init_std=projection_init_std,
        memory_mode=memory_mode,
    )
    return wrapped

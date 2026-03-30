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
        # x: [..., dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class DepthAttentionAdapter(nn.Module):
    """
    Simple full-depth adapter:
      - attends over previous hidden states
      - returns a gated additive correction
    """
    def __init__(self, hidden_size, gate_init=0.0):
        super().__init__()
        self.hidden_size = hidden_size

        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.gate = nn.Parameter(torch.tensor(gate_init)) # so that the model is intact beforing traing, which means attention residual has no effect befor training

        self.score_norm = RMSNorm(hidden_size)
        self.out_norm = RMSNorm(hidden_size)

    def forward(self, h_base, prev_states):
        """
        h_base: [B, T, D], the output from the frozen part of the model
        prev_states: list of [B, T, D], length >= 1, the hidden states of after additive PEFT styple AttnRes adapter. see more by refering to the paper
        """
        # Stack previous states: [B, Lprev, T, D]
        sources = torch.stack(prev_states, dim=1)

        # Score each source with a learned query
        normed = self.score_norm(sources)
        scores = torch.einsum("d,bltd->blt", self.query, normed)  # [B, Lprev, T]

        # Depth attention
        alpha = F.softmax(scores, dim=1)  # softmax over depth/source axis

        # Weighted sum over sources
        delta = torch.einsum("blt,bltd->btd", alpha, sources)  # [B, T, D]
        delta = self.out_norm(delta)

        # Gated residual adapter
        out = h_base + self.gate * delta
        # out = h_base # for debug



        return out, alpha



class Qwen3AttnResConfigMixin:
    """
    Optional mixin-like helper if you want to store adapter settings
    in config. You can also just pass them at init time.
    """
    attnres_lookback = None
    attnres_enable = True

class Qwen3ForCausalLMWithAttnRes(PreTrainedModel, GenerationMixin):
    """
    Wrapper around a Qwen3 causal LM with frozen backbone
    and trainable depth-attention residual adapters.
    """
    config_class = None  # set this to the actual Qwen3 config class if you want tighter integration
    base_model_prefix = "model"

    def __init__(self, config, base_model, lookback=None, gate_init=0.0):
        # This wrapper does not implement the SDPA capability checks expected by
        # PreTrainedModel, so force eager attention for safe initialization.
        if getattr(config, "_attn_implementation", None) != "eager":
            config._attn_implementation = "eager"
            config._attn_implementation_internal = "eager"
        super().__init__(config)
        self.model = base_model.model
        self.lm_head = base_model.lm_head
        self.base_lm = base_model
        self.lookback = lookback
        self.gate_init = gate_init

        hidden_size = self.model.embed_tokens.embedding_dim

        self.adapters = nn.ModuleList([
            DepthAttentionAdapter(hidden_size=hidden_size, gate_init=self.gate_init)
            for _ in range(len(self.model.layers))
        ])

        emb = self.model.embed_tokens.weight
        self.adapters.to(device=emb.device)

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

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # This wrapper currently recomputes the full forward and does not
        # maintain KV cache state across decoding steps.
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
        # Keep wrapper path by default for parity/debug checks.
        # Set strict_base_when_disabled=True only when you want hard fallback.
        force_attnres_path = kwargs.pop("force_attnres_path", True)
        strict_base_when_disabled = kwargs.pop("strict_base_when_disabled", False)

        #comment the following if you want to let the origin model and wrappered model to get the same generation. 
        # # Keep decoding defaults aligned with the wrapped base model.
        # if "generation_config" not in kwargs and hasattr(self.base_lm, "generation_config"):
        #     kwargs["generation_config"] = self.base_lm.generation_config


        # if self._adapters_effectively_disabled()
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
        **kwargs
    ):
        # Generation-only kwargs can be accidentally passed to forward() in tests.
        # Drop them to avoid leaking unexpected args into decoder layers.
        kwargs.pop("do_sample", None)

        # Keep exact base-model behavior when adapters are not active yet.
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

        # Run the exact HF backbone path and inject adapters via layer output hooks.
        # This keeps mask/cache/position handling identical to base_lm, while still
        # forcing the adapter path.
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

                if self.lookback is not None:
                    adapter_prev_states = prev_states[-self.lookback:]
                else:
                    adapter_prev_states = prev_states

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



######

# This part can be moved to a separate utils file if desired
def load_qwen3_attnres_model(
    model_name_or_path,
    lookback=8,
    gate_init=0.0,
    torch_dtype=torch.bfloat16,
    device_map="auto",
):


    """
    Helper function to load a Qwen Series model and wrap it with the AttnRes adapter.
    Autually, not only for Qwen Series, but any model with a similar architecture and HF implementation should work with minor adjustments.
    Qwen are experimented by us. 

    model_name_or_path: can be a string path(HF official name) to a pretrained model or an already loaded AutoModelForCausalLM. The function will wrap the model with the AttnRes adapter.
    lookback: can be None or an integer. If set, the adapter will only attend over the last `lookback` states instead of all previous states. This can save memory and computation with minimal impact on performance in many cases.
    gate_init: initial value for the adapter gate. Setting this to 0.0 means the adapter has no effect at the start of training, which can help with stable convergence. You can also start with a small positive value to give the adapter some initial influence.

    """
    if isinstance(model_name_or_path, str):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
    else:
        base_model = model_name_or_path

    # Ensure wrapper initialization does not request SDPA.
    base_model.config._attn_implementation = "eager"
    base_model.config._attn_implementation_internal = "eager"

    config = base_model.config
    wrapped = Qwen3ForCausalLMWithAttnRes(
        config=config,
        base_model=base_model,
        lookback=lookback,
        gate_init=gate_init,
    )
    return wrapped

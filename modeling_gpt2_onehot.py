"""Standalone GPT-2-style backbone for onehot experiments."""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GPT2Config:
    def __init__(
        self,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        scale_attn_by_inverse_layer_idx: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.use_cache = use_cache

    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def num_attention_heads(self) -> int:
        return self.n_head

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layer


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class _ModelOutput:
    __slots__ = ("last_hidden_state", "attentions")

    def __init__(self, last_hidden_state: torch.Tensor, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.attentions = attentions


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def _eager_attention_forward(module, layer_idx, query, key, value):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if layer_idx == 0:
        # Identity attention: token i attends only to itself
        T = attn_weights.shape[-1]
        attn_weights = (
            torch.eye(T, device=attn_weights.device, dtype=value.dtype)
            .unsqueeze(0).unsqueeze(0)
            .expand(query.shape[0], 1, T, T)
            .clone()
        )
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).type(value.dtype)

    attn_weights = module.attn_dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2)
    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`n_embd` ({self.embed_dim}) must be divisible by `n_head` ({self.num_heads})."
            )
        self.layer_idx = layer_idx
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

        self.q_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.register_buffer("K", torch.eye(self.embed_dim))
        self.register_buffer("V", torch.eye(self.embed_dim))

        with torch.no_grad():
            if layer_idx == 0:
                self.q_attn.weight.copy_(torch.eye(self.embed_dim))
                self.k_attn.weight.copy_(torch.eye(self.embed_dim))
                self.v_attn.weight.copy_(torch.eye(self.embed_dim))
            else:
                self.q_attn.weight.zero_()
                self.k_attn.weight.zero_()
                for i in range(4):
                    self.q_attn.weight[i, i] = 0.1
                    self.k_attn.weight[i, i] = 0.1
                self.v_attn.weight.copy_(0.1 * torch.eye(self.embed_dim))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False, **kwargs):
        B, T, _ = hidden_states.shape
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)

        q = self.q_attn(hidden_states).view(shape).transpose(1, 2)
        # uncomment for bullet 3 onehot experiments in Section 2
        # if self.layer_idx == 1:
        #     k = (hidden_states @ self.K).view(shape).transpose(1, 2)
        k = self.k_attn(hidden_states).view(shape).transpose(1, 2)
        # if self.layer_idx == 1:
        #     v = (hidden_states @ self.V).view(shape).transpose(1, 2)
        v = self.v_attn(hidden_states).view(shape).transpose(1, 2)

        attn_out, attn_weights = _eager_attention_forward(self, self.layer_idx, q, k, v)
        attn_out = attn_out.reshape(B, T, self.embed_dim).contiguous()
        attn_out = self.resid_dropout(attn_out)
        return attn_out, attn_weights


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, embed_dim: int, loss: str = "mse"):
        super().__init__()
        bound = 1.0 / math.sqrt(intermediate_size)
        self.c_fc = nn.Linear(embed_dim, intermediate_size, bias=False)
        with torch.no_grad():
            self.c_fc.weight.normal_(mean=0.0, std=bound)
        self.act = nn.ReLU()

        if loss == "mse":
            proj = torch.empty(intermediate_size, 1).uniform_(-bound, bound)
            proj = proj.to("cuda")
            self.register_buffer("c_proj", proj)
            self._trainable_proj = False
        else:
            self._proj = nn.Linear(intermediate_size, 3, bias=False)
            with torch.no_grad():
                self._proj.weight.uniform_(-bound, bound)
            self._trainable_proj = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        if self._trainable_proj:
            return self._proj(hidden_states)
        return hidden_states @ self.c_proj


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int, loss: str = "mse"):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)

        if layer_idx == 1:
            inner_dim = 4 * config.hidden_size
            self.mlp = GPT2MLP(inner_dim, config.hidden_size, loss=loss)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False, **kwargs):
        residual = hidden_states
        attn_out, attn_weights = self.attn(hidden_states, output_attentions=output_attentions)

        if self.layer_idx == 0:
            hidden_states = attn_out + residual
        else:
            hidden_states = self.mlp(attn_out)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPT2ModelOneHot(nn.Module):
    def __init__(self, config: GPT2Config, loss: str = "mse"):
        super().__init__()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2Block(config, layer_idx=i, loss=loss) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> _ModelOutput:
        hidden_states = self.drop(inputs_embeds)
        all_attentions = [] if output_attentions else None

        for block in self.h:
            block_out = block(hidden_states, output_attentions=output_attentions)
            hidden_states = block_out[0]
            if output_attentions:
                all_attentions.append(block_out[1])

        return _ModelOutput(
            last_hidden_state=hidden_states,
            attentions=tuple(all_attentions) if output_attentions else None,
        )

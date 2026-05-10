import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_gpt2_onehot import GPT2Config, GPT2ModelOneHot

try:
    from modeling_gpt2 import GPT2Model
except (TypeError, ImportError):
    GPT2Model = None  # Python 3.9+ syntax in modeling_gpt2 breaks on 3.8


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, device="cpu", loss="mse"):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions - 1,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.device = device
        self.loss = loss

        self.wte = torch.zeros(n_dims, n_embd, device=device)
        self.wte[:, :n_dims] = torch.eye(n_dims)
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)

        if loss == "softmax":
            self._read_out = nn.Linear(n_embd, 3)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs[:, :-1, :]

    def forward(self, xs, ys, zs=None, inds=None):
        if zs is not None:
            hidden = self._backbone(inputs_embeds=zs).last_hidden_state
            if self.loss == "softmax":
                return F.softmax(self._read_out(hidden), dim=-1)
            return hidden

        zs = self._combine(xs, ys)
        embeds = zs @ self.wte
        output = self._backbone(inputs_embeds=embeds).last_hidden_state

        if self.loss == "softmax":
            return F.softmax(self._read_out(output), dim=-1)[:, ::2, :]

        return output[:, ::2, 0]

    def get_embedded(self, zs=None, xs=None, ys=None, inds=None):
        if xs is not None:
            zs = self._combine(xs, ys)
        embeds = zs @ self.wte
        return zs, embeds


class TransformerModelOneHot(nn.Module):
    """Wraps the onehot backbone. xs comes pre-embedded (no _combine or wte)."""

    def __init__(self, n_embd, n_positions, n_layer, n_head, loss="mse", device="cpu"):
        super().__init__()
        config = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            use_cache=False,
        )
        self.n_dims = n_embd
        self.n_embd = n_embd
        self.loss = loss
        self._backbone = GPT2ModelOneHot(config, loss=loss)

    def forward(self, xs):
        """
        xs: (B, l, n_embd) — already in embedding space
        Returns:
          mse:                  (B, l)    scalar per position
          softmax:              (B, l, 3) softmax probabilities
          multiclass_hinge_loss:(B, l, 3) raw logits
        """
        hidden = self._backbone(inputs_embeds=xs).last_hidden_state
        if self.loss == "softmax":
            return F.softmax(hidden, dim=-1)
        if self.loss == "mse":
            return hidden[:, :, 0]
        return hidden

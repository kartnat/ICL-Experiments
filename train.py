import os
import sys

import torch
import torch.nn as nn
from quinine import QuinineArgumentParser
from tqdm import tqdm

from modeling_gpt2 import GPT2Config, GPT2Model
from models import TransformerModel
from samplers import build_all_combinations, sample_xsys_multi_signpair
from schema import schema
from tasks import mean_cross_entropy

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt_dir(base_dir, flavor, num_layers, num_heads, n_embd, n_dim, n_points):
    name = f"{flavor}_{num_layers}_{num_heads}_{n_embd}_{n_dim}_{n_points}"
    return os.path.join(base_dir, name)


def save_backbone(backbone, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))


def load_backbone(save_dir, args, device):
    config = GPT2Config(
        n_positions=2 * args.model.n_positions - 1,
        n_embd=args.model.n_embd,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=False,
    )
    backbone = GPT2Model(config)
    state = torch.load(os.path.join(save_dir, "backbone.pt"), map_location=device)
    backbone.load_state_dict(state)
    return backbone


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys.float())
    if output.dim() == 3:
        pred = output[:, -1]            # (B, 3)
        loss = loss_func(pred, ys[:, -1].long())
    else:
        loss = loss_func(output[:, -1], ys[:, -1].float())
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, args, exp_name, device):
    loss_mode = getattr(args.model, "loss", "mse")

    if loss_mode == "softmax":
        loss_func = mean_cross_entropy
    else:
        loss_func = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.training.learning_rate)

    use_fullbatch = getattr(args, "use_fullbatch", False)
    bsize    = args.training.batch_size
    n_points = args.model.n_positions
    n_embd   = args.model.n_embd
    n_dim    = args.model.n_dims

    save_dir = _ckpt_dir(
        args.out_dir, exp_name,
        args.model.n_layer, args.model.n_head, n_embd, n_dim, n_points,
    )
    os.makedirs(save_dir, exist_ok=True)

    xs_ref_path = os.path.join(save_dir, "xs.pt")
    ys_ref_path = os.path.join(save_dir, "ys.pt")

    # Pre-build full batch once (deterministic, reused every step)
    if use_fullbatch:
        dev = torch.device(device)
        sign_pair_tasks = [
            (torch.tensor([1., 1.],   device=dev), torch.tensor([-1., -1.],  device=dev), torch.tensor([-1., 1.],  device=dev)),
            (torch.tensor([1., -1.],  device=dev), torch.tensor([1., 1.],   device=dev), torch.tensor([-1., -1.], device=dev)),
            (torch.tensor([-1., 1.],  device=dev), torch.tensor([1., -1.],  device=dev), torch.tensor([1., 1.],   device=dev)),
            (torch.tensor([-1., -1.], device=dev), torch.tensor([-1., 1.],  device=dev), torch.tensor([1., -1.],  device=dev)),
        ]
        xs_fb_list, ys_fb_list = zip(*[build_all_combinations(x, y, z, total=n_points - 1) for x, y, z in sign_pair_tasks])
        xs_full = torch.cat(xs_fb_list, dim=0)
        ys_full = torch.cat(ys_fb_list, dim=0)
        xs_ref, ys_ref = xs_full, ys_full

    pbar = tqdm(range(args.training.train_steps))

    for i in pbar:
        if use_fullbatch:
            xs, ys = xs_full, ys_full
        else:
            xs, ys = sample_xsys_multi_signpair(bsize, n_points, device)
        model.train()
        loss, output = train_step(model, xs, ys, optimizer, loss_func)
        pbar.set_description(f"loss {loss:.4f}")

        save_backbone(model._backbone, save_dir)

        # Save reference xs/ys on the first iteration
        if i == 0:
            torch.save(xs, xs_ref_path)
            torch.save(ys, ys_ref_path)

        # Attention inspection using the fixed reference batch from iteration 0
        xs_ref = torch.load(xs_ref_path, map_location=device)
        ys_ref = torch.load(ys_ref_path, map_location=device)

        _, embeds = model.get_embedded(xs=xs_ref, ys=ys_ref.float())

        gpt2model = load_backbone(save_dir, args, device)
        gpt2model = gpt2model.to(device)
        gpt2model.eval()

        with torch.no_grad():
            results = gpt2model(inputs_embeds=embeds, output_attentions=True)
        attentions = results.attentions

        torch.manual_seed(42)
        rand_indices = torch.randperm(xs_ref.shape[0])[:1]
        for k in rand_indices.tolist():
            print("k input: ", xs_ref[k, :, 0:2])
            for j, a in enumerate(attentions):
                if j == 1:
                    values, indices = torch.topk(a[k, 0, -1, :], min(8, a.shape[-1]))
                    print(f"  [batch {k}] top-8 indices: {indices.tolist()}")
                    print(f"  [batch {k}] top-8 values:  {values.tolist()}")
            print("*" * 50)
        print("-" * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_mode = getattr(args.model, "loss", "mse")

    model = TransformerModel(
        n_dims=args.model.n_dims,
        n_positions=args.model.n_positions,
        n_embd=args.model.n_embd,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        loss=loss_mode,
        device=device.type,
    )

    if device.type == "cuda":
        init = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)

        V0 = torch.zeros(4, 4)
        V0[:2, :] = init

        Q = torch.zeros(4, 4)
        Q[:2, :] = init

    with torch.no_grad():
        model._backbone.h[0].attn.v_attn.weight.copy_(V0.T)
        model._backbone.h[1].attn.q_attn.weight.copy_(Q.T)

    model = model.to(device)
    model.train()
    print("device:", device)

    train(model, args, exp_name=args.exp_name, device=device.type)


if __name__ == "__main__":
    # Resolve bare config filenames against the conf/ directory
    for i, arg in enumerate(sys.argv):
        if arg.endswith(".yaml") and os.sep not in arg and "/" not in arg:
            sys.argv[i] = os.path.join("conf", arg)
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")
    main(args)

import os
import sys

import torch
import torch.nn as nn
from quinine import QuinineArgumentParser
from tqdm import tqdm

from modeling_gpt2_onehot import GPT2Config, GPT2ModelOneHot
from models import TransformerModelOneHot
from samplers import sample_xsys_multi_signpair_onehot, sample_xsys_onehot_adam
from schema import schema
from tasks import mean_cross_entropy, multiclass_hinge_loss

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


def load_backbone(save_dir, args, device, loss="mse"):
    config = GPT2Config(
        n_positions=args.model.n_positions,
        n_embd=args.model.n_embd,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=False,
    )
    backbone = GPT2ModelOneHot(config, loss=loss)
    state = torch.load(os.path.join(save_dir, "backbone.pt"), map_location=device)
    backbone.load_state_dict(state)
    return backbone


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs)
    if output.dim() == 3:
        pred = output[:, -1]            # (B, 3)
        loss = loss_func(pred, ys.long())
    else:
        loss = loss_func(output[:, -1], ys.float())
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
    elif loss_mode == "multiclass":
        loss_func = multiclass_hinge_loss
    else:
        loss_func = nn.MSELoss()

    use_adam = getattr(args, "run_altpaper_adam", False)
    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.training.learning_rate)

    n_embd   = args.model.n_embd
    bsize    = args.training.batch_size
    n_points = args.model.n_positions
    n_dim    = args.model.n_dims

    save_dir = _ckpt_dir(
        args.out_dir, exp_name,
        args.model.n_layer, args.model.n_head, n_embd, n_dim, n_points,
    )
    os.makedirs(save_dir, exist_ok=True)

    xs_ref_path = os.path.join(save_dir, "xs.pt")
    ys_ref_path = os.path.join(save_dir, "ys.pt")

    pbar = tqdm(range(args.training.train_steps))

    for i in pbar:
        if use_adam:
            xs, ys = sample_xsys_onehot_adam(bsize, n_points, device, n_embd)
        else:
            xs, ys, ts = sample_xsys_multi_signpair_onehot(
                bsize, n_points, device, D=n_embd
            )
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

        gpt2model = load_backbone(save_dir, args, device, loss=loss_mode)
        gpt2model = gpt2model.to(device)
        gpt2model.eval()

        with torch.no_grad():
            results = gpt2model(inputs_embeds=xs_ref, output_attentions=True)
        attentions = results.attentions

        for k in range(min(1, xs_ref.shape[0])):
            print("k input: ", xs_ref[k, :, 0:8])
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

    model = TransformerModelOneHot(
        n_embd=args.model.n_embd,
        n_positions=args.model.n_positions,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        loss=loss_mode,
        device=device.type,
    )
    model = model.to(device)
    model.train()
    print("device:", device)

    train(
        model, args,
        exp_name=args.exp_name,
        device=device.type,
    )


if __name__ == "__main__":
    # Resolve bare config filenames against the conf/ directory
    for i, arg in enumerate(sys.argv):
        if arg.endswith(".yaml") and os.sep not in arg and "/" not in arg:
            sys.argv[i] = os.path.join("conf", arg)
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")
    main(args)

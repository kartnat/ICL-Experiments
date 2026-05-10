import torch


def build_all_combinations(x, y, z, total=10):
    """
    For use in constructing input for full-batch gradient descent, when xs are non one-hot
    """
    device = x.device

    # all 3^10 sequences of labels {0,1,2} directly
    # generate via base-3 counting
    N = 3 ** total
    indices = torch.arange(N, device=device)  # (N,)

    # decode each position: digits[i,j] = j-th token label for sequence i
    powers = 3 ** torch.arange(total - 1, -1, -1, device=device)  # (10,)
    digits = (indices.unsqueeze(1) // powers.unsqueeze(0)) % 3    # (N, 10)

    # build data rows: digits tells us which of x,y,z to pick
    # stack x,y,z -> (3, 2), then index
    xyz = torch.stack([x, y, z], dim=0)   # (3, 2)
    data_rows = xyz[digits]               # (N, 10, 2)

    # last query row
    last = torch.zeros(N, 1, 2, device=device)
    last[:, 0, 0] = x[0]
    last[:, 0, 1] = x[1]
    xs = torch.cat([data_rows, last], dim=1)  # (N, 11, 2)

    # labels: x->2, y->1, z->0
    label_vals = torch.tensor([2, 1, 0], device=device)  # index 0->2, 1->1, 2->0
    seq_labels = label_vals[digits]                       # (N, 10)
    query_labels = torch.full((N, 1), 2,
                              dtype=torch.long, device=device)
    ys = torch.cat([seq_labels, query_labels], dim=1)    # (N, 11)

    return xs, ys

def sample_xsys_multi_signpair(B, l, device):
    """
    Minibatch version of build_all_combinations
    """
    # Sign pairs: e_1=[1,1], e_2=[1,-1], e_3=[-1,1], e_4=[-1,-1]
    basis = torch.tensor([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]], device=device)  # (4,2)

    # -------------------------------------------------
    # 1) Sample task indices (0-indexed internally)
    # -------------------------------------------------
    ts = torch.randint(0, 4, (B,), device=device)  # (B,)

    idx0 = ts % 4
    idx1 = (ts + 1) % 4
    idx2 = (ts + 2) % 4

    # -------------------------------------------------
    # 2) Sample prompt rows uniformly from {e_i, e_i+1, e_i+2}
    #    for first l-1 rows, last row is fixed to e_i+2
    # -------------------------------------------------
    rand_choice = torch.randint(0, 3, (B, l - 1), device=device)  # (B, l-1)

    idx_lookup = torch.stack([idx0, idx1, idx2], dim=1)  # (B, 3)
    sampled_basis_idx = idx_lookup.gather(1, rand_choice)  # (B, l-1)

    # Last row is always e_{i+2}
    last_idx = idx2.unsqueeze(1)  # (B, 1)
    all_basis_idx = torch.cat([sampled_basis_idx, last_idx], dim=1)  # (B, l)

    # -------------------------------------------------
    # 3) Build xs (B, l, 2)
    # -------------------------------------------------
    xs = basis[all_basis_idx]  # (B, l, 2)

    # -------------------------------------------------
    # 4) Compute labels: e_i->0, e_i+1->1, e_i+2->2
    # -------------------------------------------------
    # For each row, find which of {idx0,idx1,idx2} was sampled
    # all_basis_idx: (B, l), idx0/1/2: (B,)
    i0 = idx0.unsqueeze(1)  # (B,1)
    i1 = idx1.unsqueeze(1)
    i2 = idx2.unsqueeze(1)

    ys_all = torch.zeros(B, l, dtype=torch.long, device=device)
    ys_all[all_basis_idx == i1] = 1
    ys_all[all_basis_idx == i2] = 2

    ys = ys_all[:, -1]  # (B,) — always 2 since last row is e_{i+2}

    return xs, ys_all

def sample_xsys_multi_signpair_onehot(B, l, device, D):
    """
    Sampling the one-hot input, namely each signpair is one-hotted, but task-sampling, etc. is the same
    """
    basis = torch.tensor(
        [[1.,0., 0.,0.],[0.,1., 0.,0.],[0.,0., 1.,0.],[0.,0., 0.,1.]],
        device=device
    )  # (4,4)

    ts   = torch.randint(0, 4, (B,), device=device)
    idx0 = ts % 4
    idx1 = (ts + 1) % 4
    idx2 = (ts + 2) % 4

    rand_choice      = torch.randint(0, 3, (B, l - 1), device=device)
    idx_lookup       = torch.stack([idx0, idx1, idx2], dim=1)
    sampled_basis_idx = idx_lookup.gather(1, rand_choice)
    last_idx         = idx2.unsqueeze(1)
    all_basis_idx    = torch.cat([sampled_basis_idx, last_idx], dim=1)  # (B,l)

    xs_sign = basis[all_basis_idx]  # (B,l,4)

    i0 = idx0.unsqueeze(1)
    i1 = idx1.unsqueeze(1)
    i2 = idx2.unsqueeze(1)

    ys_all = torch.zeros(B, l, dtype=torch.long, device=device)
    ys_all[all_basis_idx == i1] = 1
    ys_all[all_basis_idx == i2] = 2

    onehot  = torch.nn.functional.one_hot(ys_all, num_classes=4).float()  # (B,l,4)

    # pad onehot to D-4 zeros so total dim = 4 (sign) + 4 (onehot) + padding = D
    pad_size = D - 4 - 4  # remaining dims
    assert pad_size >= 0, f"D={D} too small, need at least 8"
    padding = torch.zeros(B, l, pad_size, device=device)

    xs = torch.cat([xs_sign, onehot, padding], dim=-1)  # (B,l,D)

    ys = ys_all[:, -1]
    return xs, ys, ts

def sample_xsys_onehot_adam(b_size, n_points, device, n_embd,
                    n_dims_truncated=None, seeds=None):
    """
    Experiments with sampling different input to use with Adam optimizer
    """
    B, N = b_size, n_points
    assert n_embd >= 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_sign_pairs = torch.tensor([
        [ 1,  0, 0, 0],   # 0
        [ 0,  1, 0, 0],   # 1
        [ 0,  0, 1, 0],   # 2
        [ 0,  0, 0, 1],   # 3
    ], dtype=torch.float32, device=device)

    # ---- fixed label embeddings ----
    # shape: (4, n_embd-2)
    label_embeddings = torch.eye(4, n_embd - 4, device=device)

    # ---- random permutation per batch ----
    batch_mappings = torch.argsort(torch.rand(B, 4, device=device), dim=1)  # (B, 4)


    # ---- sample sign pairs ----
    rand_indices = torch.randint(0, 4, (B, N), device=device)

    # ---- build xs ----
    xs = torch.zeros((B, N, n_embd), dtype=torch.float32, device=device)

    # first two dims: sign pairs
    xs[:, :, :4] = base_sign_pairs[rand_indices]

    # mapped labels
    mapped_labels = batch_mappings.gather(1, rand_indices)  # (B, N)

    # label embeddings for all BUT last row
    xs[:, :-1, 4:] = label_embeddings[mapped_labels[:, :-1]]

    # last row embedding explicitly zero (already zero, but explicit is clearer)
    xs[:, -1, 4:] = 0.0

    # ---- ys ----
    ys = mapped_labels[:, -1]  # (B, N)

    return xs, ys

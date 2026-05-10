import torch
import torch.nn.functional as F


def multiclass_hinge_loss(outputs, ys):
    """
    outputs: (B, 3) logits
    ys:      (B,)   in {0,1,2}
    """
    ys = ys.long()
    y_onehot = F.one_hot(ys, num_classes=3).float()
    y_signed = 2 * y_onehot - 1
    loss = F.relu(1 - y_signed * outputs).sum(dim=1).mean()
    return loss

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

def mean_cross_entropy(ys_pred, ys):
    eps = 1e-9  # to avoid log(0)
    ys_onehot = torch.zeros_like(ys_pred).scatter_(-1, ys.unsqueeze(-1), 1.0)
    ce = -(ys_onehot * torch.log(ys_pred + eps)).sum(dim=-1)  # (B,)
    return ce.mean()

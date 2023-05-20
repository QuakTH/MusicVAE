import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


def ELBO_loss(
    reconstruct: torch.Tensor,
    true_x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """Calculate the minus ELBO loss.

    :param reconstruct: Reconstructed x value after decoded.
    :param true_x: Original x value.
    :param mu: Mean of the latent distribution.
    :param log_var: Log variance of the latent distribution.
    :return: minus ELBO loss.
    """
    reconstruct = torch.clip(reconstruct, 0.0, 1.0)
    likelihood = -F.binary_cross_entropy(reconstruct, true_x, reduction="none")
    likelihood = likelihood.view(likelihood.shape[0], -1).sum(1)

    sigma = torch.exp(log_var * 2)

    n_mu = torch.Tensor([0.0])
    n_sigma = torch.Tensor([1.0])

    p = Normal(n_mu, n_sigma)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)

    ELBO = torch.mean(likelihood) - torch.mean(kl_div)

    return -ELBO


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy of the one hot encoded vector.
    Check whether the predicted one hot encoding vector
    is same as the original one hot encoded vector.

    :param y_true: Original one hot encoded data.
    :param y_pred: Predicted one hot encoded data.
    :return: Accuracy.
    """
    assert y_true.shape == y_pred.shape

    _, idx_true = torch.max(y_true, dim=2)
    _, idx_pred = torch.max(y_pred, dim=2)

    return (idx_true == idx_pred).sum() / (idx_true.shape[0] * idx_true.shape[-1])

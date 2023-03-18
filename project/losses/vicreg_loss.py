import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Loss function of the VICReg method for Self-Supervised Learning
    strongly inspired from: https://github.com/untitled-ai/self_supervised
    """

    def __init__(self, invariance_loss_weight: float = 25., variance_loss_weight: float = 25., covariance_loss_weight: float = 1.):
        super(VICRegLoss, self).__init__()
        self.invariance_loss_weight = invariance_loss_weight
        self.variance_loss_weight = variance_loss_weight
        self.covariance_loss_weight = covariance_loss_weight

    def forward(self, Z_a, Z_b):
        assert Z_a.shape == Z_b.shape and len(Z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(Z_a, Z_b)

        # variance loss
        std_Z_a = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        std_Z_b = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        loss_v_a = torch.mean(F.relu(1 - std_Z_a))
        loss_v_b = torch.mean(F.relu(1 - std_Z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = Z_a.shape
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        cov_Z_a = ((Z_a.T @ Z_a) / (N - 1)).square()  # DxD
        cov_Z_b = ((Z_b.T @ Z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_Z_a.sum() - cov_Z_a.diagonal().sum()) / D
        loss_c_b = (cov_Z_b.sum() - cov_Z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.invariance_loss_weight
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss

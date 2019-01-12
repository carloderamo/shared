import torch
import torch.nn.functional as F


class LossFunction(object):
    def __init__(self, reg_coeff):
        self._reg_coeff = reg_coeff

    def __call__(self, arg, y):
        raise NotImplementedError


class FeaturesL1Loss(LossFunction):
    def __init__(self, reg_coeff):
        super().__init__(reg_coeff)

    def __call__(self, arg, y):
        yhat, h_f = arg

        loss = F.mse_loss(yhat, y)

        l1_loss = torch.norm(h_f, 1, dim=1)
        l1_loss = torch.mean(l1_loss)

        return loss + self._reg_coeff * l1_loss


class FeaturesKLLoss(LossFunction):
    def __init__(self, k, reg_coeff):
        self._k = k
        super().__init__(reg_coeff)

    def __call__(self, arg, y):
        yhat, h_f = arg

        loss = F.mse_loss(yhat, y)

        mu_s = torch.sum(h_f, dim=1)
        kl_loss = -self._k * torch.log(mu_s) + mu_s
        kl_loss = torch.mean(kl_loss)

        return loss + self._reg_coeff * kl_loss
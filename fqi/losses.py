import torch
import torch.nn.functional as F


class LossFunction(object):
    def __init__(self, n_games):
        self._n_games = n_games

        self._losses = list()
        self._counter = 0

    def get_losses(self):
        return self._losses

    def __call__(self, yhat, y):
        loss = F.mse_loss(yhat, y, reduce=True)

        return loss

    def _need_log(self):
        self._counter += 1
        if self._counter >= self._eval_frequency:
            self._counter = 0
            return True
        else:
            return False

import torch
import torch.nn.functional as F


class LossFunction(object):
    def __init__(self, reg_coeff, n_games, batch_size, eval_frequency):
        self._reg_coeff = reg_coeff
        self._n_games = n_games
        self._batch_size = batch_size
        self._eval_frequency = eval_frequency

        self._losses = list()
        self._reg_losses = list()
        self._counter = 0

    def get_losses(self):
        return self._losses

    def get_reg_losses(self):
        return self._reg_losses

    def __call__(self, yhat, y):
        loss = F.smooth_l1_loss(yhat, y, reduce=False)

        if self._need_log():
            temp_losses = list()
            temp_l1_losses = list()

            for i in range(self._n_games):
                start = i * self._batch_size
                stop = start + self._batch_size
                temp_losses.append(torch.mean(loss[start:stop]).item())

            self._losses.append(temp_losses)
            self._reg_losses.append(temp_l1_losses)

        loss = torch.mean(loss)

        return loss

    def _need_log(self):
        self._counter += 1
        if self._counter >= self._eval_frequency:
            self._counter = 0
            return True
        else:
            return False

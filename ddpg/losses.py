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

    def __call__(self, arg, y):
        yhat, h_f = arg

        return F.smooth_l1_loss(yhat, y)

    def _need_log(self):
        self._counter += 1
        if self._counter >= self._eval_frequency:
            self._counter = 0
            return True
        else:
            return False


class FeaturesL1Loss(LossFunction):
    def __init__(self, reg_coeff, n_games, batch_size, eval_frequency):
        super().__init__(reg_coeff, n_games, batch_size, eval_frequency)

    def __call__(self, arg, y):
        yhat, h_f = arg

        loss = F.smooth_l1_loss(yhat, y, reduce=False)
        l1_loss = torch.norm(h_f, 1, dim=1)

        if self._need_log():
            temp_losses = list()
            temp_l1_losses = list()

            for i in range(self._n_games):
                start = i * self._batch_size
                stop = start + self._batch_size
                temp_losses.append(torch.mean(loss[start:stop]).item())
                temp_l1_losses.append(torch.mean(l1_loss[start:stop]).item())

            self._losses.append(temp_losses)
            self._reg_losses.append(temp_l1_losses)

        loss = torch.mean(loss)
        l1_loss = torch.mean(l1_loss)

        return loss + self._reg_coeff * l1_loss


class FeaturesKLLoss(LossFunction):
    def __init__(self, k, reg_coeff, n_games, batch_size, eval_frequency):
        self._k = k
        super().__init__(reg_coeff, n_games, batch_size, eval_frequency)

    def __call__(self, arg, y):
        yhat, h_f = arg

        loss = F.smooth_l1_loss(yhat, y, reduce=False)
        mu_s = torch.sum(h_f, dim=1)

        kl_loss = -self._k * torch.log(mu_s) + mu_s

        if self._need_log():
            temp_losses = list()
            temp_kl_losses = list()
            for i in range(self._n_games):
                start = i * self._batch_size
                stop = start + self._batch_size
                temp_losses.append(torch.mean(loss[start:stop]).item())
                temp_kl_losses.append(torch.mean(kl_loss[start:stop]).item())
            self._losses.append(temp_losses)
            self._reg_losses.append(temp_kl_losses)

        loss = torch.mean(loss)
        kl_loss = torch.mean(kl_loss)

        return loss + self._reg_coeff * kl_loss


class WeightsL1Loss(LossFunction):
    def __init__(self, n_actions_per_head, reg_coeff, n_games,
                 batch_size, eval_frequency):
        self._n_actions_per_head = n_actions_per_head
        super().__init__(reg_coeff, n_games, batch_size, eval_frequency)

    def __call__(self, arg, y):
        yhat, w = arg

        loss = F.smooth_l1_loss(yhat, y, reduce=False)

        temp_losses = list()
        temp_l1_losses = list()
        l1_loss = list()

        for i in range(self._n_games):
            start = i * self._batch_size
            stop = start + self._batch_size
            temp_losses.append(torch.mean(loss[start:stop]).item())

            tmp = torch.norm(w[i].weight[:self._n_actions_per_head[i][0]], 1)
            l1_loss.append(tmp)
            temp_l1_losses.append(tmp.item())

        if self._need_log():
            self._losses.append(temp_losses)
            self._reg_losses.append(temp_l1_losses)

        loss = torch.mean(loss)
        l1_loss = torch.mean(torch.Tensor(l1_loss))

        return loss + self._reg_coeff * l1_loss


class WeightsGLLoss(LossFunction):
    def __init__(self, n_actions_per_head, reg_coeff, n_games,
                 batch_size, eval_frequency):
        self._n_actions_per_head = n_actions_per_head
        super().__init__(reg_coeff, n_games, batch_size, eval_frequency)

    def __call__(self, arg, y):
        yhat, w = arg

        loss = F.smooth_l1_loss(yhat, y, reduce=False)

        temp_losses = list()
        w = [x.weight[:self._n_actions_per_head[i][0]] for i, x in enumerate(w)]
        w = torch.cat(w)
        for i in range(self._n_games):
            start = i * self._batch_size
            stop = start + self._batch_size
            temp_losses.append(torch.mean(loss[start:stop]).item())

        gl1_loss = torch.norm(torch.norm(w, 2, dim=0), 1)

        if self._need_log():
            self._losses.append(temp_losses)
            self._reg_losses.append(gl1_loss.item())

        loss = torch.mean(loss)

        return loss + self._reg_coeff * gl1_loss

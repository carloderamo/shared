import torch

from mushroom.approximators.parametric.pytorch_network import PyTorchApproximator


class CustomPyTorchApproximator(PyTorchApproximator):
    def __init__(self, n_actions_per_head, **pars):
        super().__init__(n_actions_per_head=n_actions_per_head, **pars)

        self.grad = 0.

    def _fit_batch(self, batch, use_weights, kwargs):
        params = kwargs.pop('params', None)

        loss = self._compute_batch_loss(batch, use_weights, kwargs)

        if params is None:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        else:
            self._optimizer.zero_grad()
            loss.backward()
            self.grad = 0.
            for pars in params:
                for p in pars:
                    self.grad += torch.norm(p.grad.data)

        return loss.item()

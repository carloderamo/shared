import numpy as np
import torch

from mushroom.approximators.parametric.pytorch_network import PyTorchApproximator


class CustomPyTorchApproximator(PyTorchApproximator):
    def __init__(self, n_actions_per_head, **pars):
        super().__init__(n_actions_per_head=n_actions_per_head, **pars)
        self.grads = np.zeros(len(n_actions_per_head))

    def _fit_batch(self, batch, use_weights, kwargs):
        params = kwargs.pop('params', None)
        idx = kwargs.pop('er_idx', None)

        loss = self._compute_batch_loss(batch, use_weights, kwargs)

        if params is None:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        else:
            for pars in params:
                for p in pars:
                    g = torch.autograd.grad(loss, p, retain_graph=True)[0]
                    self.grads[idx] = torch.norm(g)

        return loss.item()

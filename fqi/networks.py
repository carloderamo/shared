import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LQRNetwork(nn.Module):
    def __init__(self, input_shape, _, use_cuda, features,
                 dropout, n_features=5):
        super(LQRNetwork, self).__init__()

        self._n_input = input_shape
        self._n_games = len(self._n_input)
        self._use_cuda = use_cuda
        self._n_shared = 2
        self._features = features

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], n_features) for i in range(
                len(input_shape))]
        )
        self._h2 = nn.Linear(n_features, n_features)
        self._q = nn.ModuleList(
            [nn.Linear(n_features, 1) for _ in range(
                self._n_games)]
        )

        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        for i in range(self._n_games):
            nn.init.xavier_uniform_(self._h1[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._q[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, idx=None):
        state = state.float()

        h1 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1.append(F.relu(self._h1[i](state[idxs, :self._n_input[i][0]])))
        cat_h1 = torch.cat(h1)

        if self._features == 'relu':
            h_f = F.relu(self._h2(cat_h1))
        elif self._features == 'sigmoid':
            h_f = torch.sigmoid(self._h2(cat_h1))
        else:
            raise ValueError

        q = [self._q[i](h_f) for i in range(self._n_games)]
        q = torch.stack(q, dim=1)

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            if q.dim() == 2:
                q_idx = q.gather(1, idx.unsqueeze(-1))
            else:
                q_idx = q.gather(1, idx.view(-1, 1).repeat(
                    1, 1).unsqueeze(1))

            q = torch.squeeze(q_idx, 1)

        return q

    def get_shared_weights(self):
        p2 = list()

        for p in self._h2.parameters():
            p2.append(p.data.detach().cpu().numpy())

        return p2

    def set_shared_weights(self, weights):
        w2 = weights

        for p, w in zip(self._h2.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            if self._use_cuda:
                w_tensor = w_tensor.cuda()
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = True

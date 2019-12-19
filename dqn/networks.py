import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AtariNetwork(nn.Module):
    n_features = 512

    def __init__(self, input_shape, _, n_actions_per_head, use_cuda, n_games,
                 features, dropout):
        super().__init__()

        self._n_input = input_shape
        self._n_games = n_games
        self._max_actions = max(n_actions_per_head)[0]
        self._features = features
        self._use_cuda = use_cuda
        self._n_shared = 2

        self._h1 = nn.ModuleList(
            [nn.Conv2d(self._n_input[0], 32, kernel_size=8, stride=4) for _ in range(
                self._n_games)]
        )
        self._h2 = nn.ModuleList(
            [nn.Conv2d(32, 64, kernel_size=4, stride=2) for _ in range(
                self._n_games)]
        )
        self._h3 = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1) for _ in range(
                self._n_games)]
        )
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.ModuleList(
            [nn.Linear(self.n_features, self._max_actions) for _ in range(
                self._n_games)]
        )

        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        for i in range(self._n_games):
            nn.init.xavier_uniform_(self._h1[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h2[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h3[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h5[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, idx=None):
        state = state.float() / 255.

        h = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h_f = F.relu(
                self._h1[i](state[idxs, :self._n_input[0]])
            )
            h_f = F.relu(self._h2[i](h_f))
            h.append(F.relu(self._h3[i](h_f)))
        cat_h3 = torch.cat(h)

        if self._features == 'relu':
            h_f = F.relu(self._h4(cat_h3.view(-1, 3136)))
        elif self._features == 'sigmoid':
            h_f = torch.sigmoid(self._h4(cat_h3.view(-1, 3136)))
        else:
            raise ValueError

        q = [self._h5[i](h_f) for i in range(self._n_games)]
        q = torch.stack(q, dim=1)

        if action is not None:
            action = action.long()
            q_acted = torch.squeeze(
                q.gather(2, action.repeat(1, self._n_games).unsqueeze(-1)), -1)

            q = q_acted

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            if q.dim() == 2:
                q_idx = q.gather(1, idx.unsqueeze(-1))
            else:
                q_idx = q.gather(1, idx.view(-1, 1).repeat(
                    1, self._max_actions).unsqueeze(1))

            q = torch.squeeze(q_idx, 1)

        return q

    def get_shared_weights(self):
        p1 = list()

        for p in self._h4.parameters():
            p1.append(p.data.detach().cpu().numpy())

        return p1

    def set_shared_weights(self, weights):
        w1 = weights

        for p, w in zip(self._h4.parameters(), w1):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            if self._use_cuda:
                w_tensor = w_tensor.cuda()
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h4.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h4.parameters():
            p.requires_grad = True


class GymNetwork(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, use_cuda, features,
                 dropout, n_features=80):
        super().__init__()

        self._n_input = input_shape
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._use_cuda = use_cuda
        self._n_shared = 4
        self._features = features

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], n_features) for i in range(
                len(input_shape))]
        )
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.ModuleList(
            [nn.Linear(n_features, self._max_actions) for _ in range(
                self._n_games)]
        )

        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        for i in range(self._n_games):
            nn.init.xavier_uniform_(self._h1[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h4[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, idx=None):
        state = state.float()

        h1 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1.append(F.relu(self._h1[i](state[idxs, :self._n_input[i][0]])))
        cat_h1 = torch.cat(h1)

        h_f = F.relu(self._h2(cat_h1))

        if self._features == 'relu':
            h_f = F.relu(self._h3(h_f))
        elif self._features == 'sigmoid':
            h_f = torch.sigmoid(self._h3(h_f))
        else:
            raise ValueError

        q = [self._h4[i](h_f) for i in range(self._n_games)]
        q = torch.stack(q, dim=1)

        if action is not None:
            action = action.long()
            q_acted = torch.squeeze(
                q.gather(2, action.repeat(1, self._n_games).unsqueeze(-1)), -1)

            q = q_acted

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            if q.dim() == 2:
                q_idx = q.gather(1, idx.unsqueeze(-1))
            else:
                q_idx = q.gather(1, idx.view(-1, 1).repeat(
                    1, self._max_actions).unsqueeze(1))

            q = torch.squeeze(q_idx, 1)

        return q

    def get_shared_weights(self):
        p2 = list()
        p3 = list()

        for p in self._h2.parameters():
            p2.append(p.data.detach().cpu().numpy())

        for p in self._h3.parameters():
            p3.append(p.data.detach().cpu().numpy())

        return p2, p3

    def set_shared_weights(self, weights):
        w2, w3 = weights

        for p, w in zip(self._h2.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            if self._use_cuda:
                w_tensor = w_tensor.cuda()
            p.data = w_tensor

        for p, w in zip(self._h3.parameters(), w3):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            if self._use_cuda:
                w_tensor = w_tensor.cuda()
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = False
        for p in self._h3.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = True
        for p in self._h3.parameters():
            p.requires_grad = True

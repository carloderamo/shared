import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, n_hidden_1,
                 n_hidden_2, use_cuda, features, dropout):
        super().__init__()

        self._n_input = input_shape
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._use_cuda = use_cuda
        self._features = features
        self._n_shared = 2

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], n_hidden_1) for i in range(
                len(input_shape))]
        )
        self._h2 = nn.Linear(n_hidden_1, n_hidden_2)
        self._h3 = nn.ModuleList(
            [nn.Linear(n_hidden_2, self._max_actions) for _ in range(
                self._n_games)]
        )

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h2.weight)
        nn.init.uniform_(self._h2.weight, a=-1 / np.sqrt(fan_in),
                         b=1 / np.sqrt(fan_in))
        for i in range(self._n_games):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h1[i].weight)
            nn.init.uniform_(self._h1[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            nn.init.uniform_(self._h3[i].weight, a=-3e-3, b=3e-3)
            nn.init.uniform_(self._h3[i].bias, a=-3e-3, b=3e-3)

    def forward(self, state, idx=None, get_features=False):
        state = state.float()

        h1 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1.append(F.relu(self._h1[i](state[idxs, :self._n_input[i][0]])))
        cat_h1 = torch.cat(h1)

        h_f = F.relu(self._h2(cat_h1))

        a = [torch.tanh(self._h3[i](h_f)) for i in range(self._n_games)]
        a = torch.stack(a, dim=1)

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            a_idx = a.gather(1, idx.view(-1, 1).repeat(
                1, self._max_actions).unsqueeze(1)
                )

            a = torch.squeeze(a_idx, 1)

        if get_features:
            return a, h_f
        else:
            return a

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


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, n_hidden_1,
                 n_hidden_2, use_cuda, features, dropout):
        super().__init__()

        self._n_input = input_shape
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._n_actions_per_head = n_actions_per_head
        self._use_cuda = use_cuda
        self._features = features
        self._n_shared = 2

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], n_hidden_1) for i in range(
                len(input_shape))]
        )
        self._h2_s = nn.Linear(n_hidden_1, n_hidden_2)
        self._h3 = nn.ModuleList(
            [nn.Linear(n_hidden_2, 1) for _ in range(
                self._n_games)]
        )
        self._h2_a = nn.ModuleList(
            [nn.Linear(n_actions_per_head[i][0], n_hidden_2, bias=False) for i in range(
                len(n_actions_per_head))]
        )

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_s.weight)
        nn.init.uniform_(self._h2_s.weight, a=-1 / np.sqrt(fan_in),
                         b=1 / np.sqrt(fan_in))
        for i in range(self._n_games):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self._h2_a[i].weight)
            nn.init.uniform_(self._h2_a[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h1[i].weight)
            nn.init.uniform_(self._h1[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            nn.init.uniform_(self._h3[i].weight, a=-3e-3, b=3e-3)
            nn.init.uniform_(self._h3[i].bias, a=-3e-3, b=3e-3)

    def forward(self, state, action, idx=None):
        state = state.float()
        action = action.float()
        if not isinstance(idx, np.ndarray):
            idx = idx.cpu().numpy().astype(np.int)

        h2 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1 = F.relu(self._h1[i](state[idxs, :self._n_input[i][0]]))
            a = action[idxs, :self._n_actions_per_head[i][0]]
            h2.append(self._h2_s(h1) + self._h2_a[i](a))

        cat_h2 = torch.cat(h2)

        if self._features == 'relu':
            h_f = F.relu(cat_h2)
        elif self._features == 'sigmoid':
            h_f = torch.sigmoid(cat_h2)
        else:
            raise ValueError

        q = [self._h3[i](h_f) for i in range(self._n_games)]
        q = torch.stack(q, dim=1).squeeze(-1)

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()

            q_idx = q.gather(1, idx.unsqueeze(-1))
            q = torch.squeeze(q_idx, 1)

        return q

    def get_shared_weights(self):
        p2 = list()

        for p in self._h2_s.parameters():
            p2.append(p.data.detach().cpu().numpy())

        return p2

    def set_shared_weights(self, weights):
        w2 = weights

        for p, w in zip(self._h2_s.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            if self._use_cuda:
                w_tensor = w_tensor.cuda()
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2_s.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2_s.parameters():
            p.requires_grad = True

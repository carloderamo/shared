import numpy as np


def step(mdp, state, action):
    mdp.reset(state)

    return mdp.step(action)


def bfs(mdp, frontier, k, max_k):
    if len(frontier) == 0 or k == max_k:
        return k

    new_frontier = list()
    for f in frontier:
        s, r, _, _ = step(mdp, f, [0])
        if r == 1:
            return k
        elif r == 0:
            new_frontier.append(s)

        s, r, _, _ = step(mdp, f, [1])
        if r == 1:
            return k
        elif r == 0:
            new_frontier.append(s)

        print(k)

    return bfs(mdp, new_frontier, k + 1, max_k)


def solve_car_on_hill(mdp, states, actions, gamma, max_k=50):
    q = list()
    k = list()
    for s, a in zip(states, actions):
        mdp.reset(s)
        state, reward, _, _ = mdp.step(a)

        if reward == 1:
            k = 1
        elif reward == -1:
            k = 1
        else:
            k = bfs(mdp, [state], 1, max_k)

    print(k)

    exit()

    return q

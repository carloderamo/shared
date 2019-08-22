import numpy as np


def step(mdp, state, action):
    mdp.reset(state)

    return mdp.step(action)


def bfs(mdp, frontier, k):
    new_frontier = list()
    for f in frontier:
        s, r0, _, _ = step(mdp, f, [0])
        new_frontier.append(s)
        s, r1, _, _ = step(mdp, f, [1])
        new_frontier.append(s)

        print(r0, r1, k)

        if r0 == 1 or r1 == 1:
            return k

    return bfs(mdp, new_frontier, k + 1)


def solve_car_on_hill(mdp, states, actions, gamma):
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
            k = bfs(mdp, [state], 1)

    print(k)

    exit()

    return q

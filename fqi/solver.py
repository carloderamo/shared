def step(mdp, state, action):
    mdp.reset(state)

    return mdp.step(action)


def bfs(mdp, frontier, k, max_k):
    if len(frontier) == 0 or k == max_k:
        return False, k

    new_frontier = list()
    for f in frontier:
        s, r, _, _ = step(mdp, f, [0])
        if r == 1:
            return True, k
        elif r == 0:
            new_frontier.append(s)

        s, r, _, _ = step(mdp, f, [1])
        if r == 1:
            return True, k
        elif r == 0:
            new_frontier.append(s)

    return bfs(mdp, new_frontier, k + 1, max_k)


def solve_car_on_hill(mdp, states, actions, gamma, max_k=50):
    q = list()
    for s, a in zip(states, actions):
        mdp.reset(s)
        state, reward, _, _ = mdp.step(a)

        if reward == 1:
            k = 1
            success = True
        elif reward == -1:
            k = 1
            success = False
        else:
            success, k = bfs(mdp, [state], 2, max_k)

        if success:
            q.append(gamma ** (k - 1))
        else:
            q.append(-gamma ** (k - 1))

    return q

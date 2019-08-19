import numpy as np


def computeOptimalK(A, B, Q, R, gamma):
    """
    This function computes the optimal linear controller associated to the
    LQG problem (u = K * x).

    Returns:
        K (matrix): the optimal controller

    """
    P = np.eye(Q.shape[0], Q.shape[1])
    for i in range(100):
        K = -gamma * np.dot(np.linalg.inv(
            R + gamma * (np.dot(B.T, np.dot(P, B)))),
            np.dot(B.T, np.dot(P, A)))
        P = computeP2(K, A, B, Q, R, gamma)
    K = -gamma * np.dot(np.linalg.inv(R + gamma * (np.dot(B.T, np.dot(P, B)))),
                        np.dot(B.T, np.dot(P, A)))

    return K


def computeP2(K, A, B, Q, R, gamma):
    """
    This function computes the Riccati equation associated to the LQG
    problem.

    Args:
        K (matrix): the matrix associated to the linear controller K * x

    Returns:
        P (matrix): the Riccati Matrix

    """
    I = np.eye(Q.shape[0], Q.shape[1])
    if np.array_equal(A, I) and np.array_equal(B, I):
        P = (Q + np.dot(K.T, np.dot(R, K))) / (I - gamma * (I + 2 * K + K ** 2))
    else:
        tolerance = .0001
        converged = False
        P = np.eye(Q.shape[0], Q.shape[1])
        while not converged:
            Pnew = Q + gamma * np.dot(A.T, np.dot(P, A)) + \
                gamma * np.dot(K.T, np.dot(B.T, np.dot(P, A))) + \
                gamma * np.dot(A.T, np.dot(P, np.dot(B, K))) + \
                gamma * np.dot(K.T, np.dot(B.T, np.dot(P, np.dot(B, K)))) + \
                np.dot(K.T, np.dot(R, K))
            converged = np.max(np.abs(P - Pnew)) < tolerance
            P = Pnew

    return P


def computeQFunction(x, u, K, A, B, Q, R, Sigma, gamma, n_random_xn=100):
    """
    This function computes the Q-value of a pair (x,u) given the linear
    controller Kx + epsilon where epsilon \sim N(0, Sigma).

    Args:
        x (int, array): the state
        u (int, array): the action
        K (matrix): the controller matrix
        Sigma (matrix): covariance matrix of the zero-mean noise added to
        the controller action
        n_random_xn: the number of samples to draw in order to average over
        the next state

    Returns:
        Qfun (float): The Q-value in the given pair (x,u) under the given
        controller

    """
    P = computeP2(K, A, B, Q, R, gamma)
    Qfun = 0
    for i in range(n_random_xn):
        action_noise = np.random.multivariate_normal(
            np.zeros(Sigma.shape[0]), Sigma, 1)
        nextstate = np.dot(A, x) + np.dot(B, u + action_noise)
        Qfun -= np.dot(x.T, np.dot(Q, x)) + \
            np.dot(u.T, np.dot(R, u)) + \
            gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
            (gamma / (1 - gamma)) * \
            np.trace(np.dot(Sigma, R + gamma * np.dot(B.T, np.dot(P, B))))
    Qfun = Qfun.item() / n_random_xn

    return Qfun

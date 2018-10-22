import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval
    
games = ['cart.npy', 'acro.npy', 'mc.npy', 'coh.npy', 'pend.npy']
n_games = len(games)

plt.suptitle('Single VS SingleREG')

for i, g in enumerate(games):
    if g == '':
        continue

    plt.subplot(n_games, 1, i + 1)
    plt.title(g)
    
    game = np.load('multi/' + g)[:, 0]
    game_mean_1, game_err_1 = get_mean_and_confidence(game)
    
    game = np.load('multireg1e-4/' + g)[:, 0]
    game_mean_2, game_err_2 = get_mean_and_confidence(game)
    
    plt.plot(game_mean_1)
    plt.fill_between(np.arange(51), game_mean_1- game_err_1, game_mean_1 + game_err_1, alpha=.5)
    
    plt.plot(game_mean_2)
    plt.fill_between(np.arange(51), game_mean_2 - game_err_2, game_mean_2 + game_err_2, alpha=.5)
    
    plt.grid()
    
plt.legend(['Single', 'SingleREG'])
    
plt.show()
    

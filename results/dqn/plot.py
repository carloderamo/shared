import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

folders = ['dqn', 'multidqn']
settings = ['noreg', 'reg1e-4', 'reg1e-4sigmoid']
games = ['cart.npy', 'acro.npy', 'mc.npy', 'coh.npy', 'pend.npy']
n_games = len(games)
n_settings = len(settings)

plt.suptitle('DQN VS MULTI')

for i, g in enumerate(games):
    if g == '':
        continue
    for j, s in enumerate(settings):
        plt.subplot(n_games, n_settings, i * n_settings + j + 1)
        plt.title(g + '-' + s)
        
        game = np.load('dqn/' + s + '/' + g)[:, 0]
        game_mean, game_err = get_mean_and_confidence(game)
        
        multi_game = np.load('multidqn/' + s + '.npy')[:, i]
        multi_game_mean, multi_game_err = get_mean_and_confidence(multi_game)
        
        plt.plot(game_mean)
        plt.fill_between(np.arange(51), game_mean - game_err, game_mean + game_err, alpha=.5)
        
        plt.plot(multi_game_mean)
        plt.fill_between(np.arange(51), multi_game_mean - multi_game_err, multi_game_mean + multi_game_err, alpha=.5)
    
        plt.grid()
    
plt.legend(['DQN', 'MULTI'])
    
plt.show()
    

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

multi = np.load('multi/multi.npy')
multi_reg = np.load('multireg1e-4sigmoid/multi.npy')

plt.suptitle('NO REG VS REG')

for i, g in enumerate(games):
    if g == '':
        continue

    plt.subplot(n_games, 1, i + 1)
    plt.title(g)

    multi_game = multi[:, i]
    multi_game_mean, multi_game_err = get_mean_and_confidence(multi_game)
    
    plt.plot(multi_game_mean)
    plt.fill_between(np.arange(51), multi_game_mean - multi_game_err, multi_game_mean + multi_game_err, alpha=.5)
    
    multi_reg_game = multi_reg[:, i]
    multi_reg_game_mean, multi_reg_game_err = get_mean_and_confidence(multi_reg_game)
    
    plt.plot(multi_reg_game_mean)
    plt.fill_between(np.arange(51), multi_reg_game_mean - multi_reg_game_err, multi_reg_game_mean + multi_reg_game_err, alpha=.5)
    
    plt.grid()
    
plt.legend(['NO REG', 'REG'])
    
plt.show()
    


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
games = ['cart', 'acro', 'mc', 'coh', 'pend']
reg = ['noreg', 'l1', 'l1-weights']
activation = ['relu', 'sigmoid']
n_games = len(games)

plt.suptitle('DQN VS MULTI')

for i, g in enumerate(games):
    if g == '':
        continue

    plt.subplot(n_games, 1, i + 1)
    plt.title(g)

    best_single_regret = np.inf
    best_multi_regret = np.inf
    single_list = list()
    multi_list = list()
    settings = list()
    
    for act in activation:
        for r in reg:
            path = r + '-' + act
            settings.append(path)
            try:
                single_list.append(np.load('dqn/' + path + '/scores.npy')[:, i])
            except:
                single_list.append(np.ones((100, 51)) * -np.inf)
            multi_list.append(np.load('multidqn/' + path + '/scores.npy')[:, i])

    singles = np.array(single_list)
    multis = np.array(multi_list)

    max_single = singles.mean(1).max()
    single_regret = (max_single - singles.mean(1)).sum(-1)
    best_single = np.argmin(single_regret)

    max_multi = multis.mean(1).max()
    multi_regret = (max_multi - multis.mean(1)).sum(-1)
    best_multi = np.argmin(multi_regret)
            
    game = np.load('dqn/' + settings[best_single] + '/scores.npy')[:, i]
    multi_game = np.load('multidqn/' + settings[best_multi] + '/scores.npy')[:, i]
    
    print('Single: ' + settings[best_single])
    print('Multi: ' + settings[best_multi])
            
    game_mean, game_err = get_mean_and_confidence(game)       
    multi_game_mean, multi_game_err = get_mean_and_confidence(multi_game)
    
    plt.plot(game_mean)
    plt.fill_between(np.arange(51), game_mean - game_err, game_mean + game_err, alpha=.5)
    
    plt.plot(multi_game_mean)
    plt.fill_between(np.arange(51), multi_game_mean - multi_game_err, multi_game_mean + multi_game_err, alpha=.5)

    plt.grid()
    
plt.legend(['DQN', 'MULTI'])
    
plt.show()
    

import numpy as np


def preprocess(dataset, evaluation_frequency=1000):
    n_steps = dataset.shape[1] // evaluation_frequency
    
    if dataset.ndim == 3:
        prepro_dataset = np.zeros((dataset.shape[0], n_steps, dataset.shape[2]))
    else:
        prepro_dataset = np.zeros((dataset.shape[0], n_steps))
    for i in range(n_steps):
        start = i * evaluation_frequency
        stop = start + evaluation_frequency
        prepro_dataset[:, i] = dataset[:, start:stop].mean(1)

    return np.array(prepro_dataset)
    
    
alg = 'multidqn'
games = ['cart', 'acro', 'mc', 'coh', 'pend']
reg = ['noreg', 'l1']
activation = ['relu', 'sigmoid']
files = ['loss', 'l1_loss', 'v']

for act in activation:
    for r in reg:
        for f in files:
            title = r + '-' + act
            path = alg + '/' + title + '/'
            a = np.load(path + f + '_raw.npy')
            a = preprocess(a)
            np.save(path + f + '.npy', a)


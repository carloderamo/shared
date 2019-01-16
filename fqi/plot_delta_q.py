import pickle

from matplotlib import pyplot as plt
import numpy as np

path = 'logs/'
folder_name = ['batch_gym_2019-01-15_15-53-14/', 'batch_gym_2019-01-15_15-53-17/']

args = pickle.load(open(path + folder_name[0] + 'args.pkl', 'rb'))

exp = 0

fig, ax = plt.subplots(2, 1)
for f in folder_name:
    delta_q = list()
    q_list = list()
    q_old = 0.
    for i in range(args.n_iterations):
        q_current = np.load(path + f + 'targets-exp-%d-%d.npy' % (exp, i))
        delta_q.append(q_old - q_current)

        q_old = q_current

    delta_q = np.array(delta_q)

    ax[0].plot(np.linalg.norm(delta_q, axis=1))
    ax[1].plot(np.linalg.norm(delta_q, axis=1, ord=np.inf))

plt.show()

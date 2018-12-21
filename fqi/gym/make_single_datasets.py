import pickle


folder = '0.5'

mdps = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'caronhill', 'pendulum']

with open(folder + '/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

for i, m in enumerate(mdps):
    current_dataset = dataset[i::len(mdps)]
    for d in current_dataset:
        d[0][0] = 0

    with open(folder + '/' + m + '.pkl', 'wb') as f:
        pickle.dump(current_dataset, f)

import numpy as np


def minibatch_number(size, batch_size):
    """
    Function to retrieve the number of batches, given a batch sizes.

    Args:
        size (int): size of the dataset;
        batch_size (int): size of the batches.

    Returns:
        The number of minibatches in the dataset.

    """
    return int(np.ceil(size / batch_size))


def reorder_data(data, idxs, games):
    new_data = list()
    for g in games:
        new_data.append(data[np.argwhere(idxs == g).ravel()])

    new_data = np.concatenate(new_data, axis=0)

    return new_data


def multi_minibatch_generator(batch_size, *dataset):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        dataset: the dataset to be splitted.

    Returns:
        The current minibatch.

    """
    size = len(dataset[0])
    num_batches = minibatch_number(size, batch_size)
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            batch.append(dataset[i][indexes[batch_start:batch_end]])

        idxs = batch[-2].ravel()

        games = np.unique(idxs)

        ordered_batch = list()

        for batch_i in batch:
            ordered_batch.append(reorder_data(batch_i, idxs, games))

        yield ordered_batch

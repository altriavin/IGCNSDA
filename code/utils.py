import numpy as np
import torch
from dataloader import GetData

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def UniformSample(dataset:GetData):
    snoRNAs = np.random.randint(0, dataset.n_snoRNA, dataset.trainSize)
    allPos = dataset.allPos
    S = []
    for snoRNA in snoRNAs:
        posForSnoRNA = allPos[snoRNA]
        if len(posForSnoRNA) == 0:
            continue

        posdisease = posForSnoRNA[np.random.randint(0, len(posForSnoRNA))]
        negdisease = np.random.randint(0, dataset.m_disease)
        while negdisease in posForSnoRNA:
            negdisease = np.random.randint(0, dataset.m_disease)

        S.append([snoRNA, posdisease, negdisease])

    return np.array(S)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def generate_batches(tensors, batch_size):
    for i in range(0, len(tensors), batch_size):
        yield tensors[i:i + batch_size]

def minibatch(tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim,:]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())
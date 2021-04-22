import torch
import numpy as np


def array_to_bits(matrix, bits, n_ints = None):
    if n_ints is not None:
        matrix = (matrix * n_ints).astype(int)
    n_samples, n_features = matrix.shape
    get_bin = lambda n, z: np.array(list(format(n, 'b').zfill(z))).astype(int)
    x_train_bits = np.stack([get_bin(num, bits) for num_list in matrix for num in num_list])
    return torch.FloatTensor(x_train_bits).view(n_samples, n_features, bits)

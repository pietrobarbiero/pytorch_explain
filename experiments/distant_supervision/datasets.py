import random
import typing
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import List, Tuple, Generator, Union

import torch
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from  torch.utils.data import TensorDataset, DataLoader
from itertools import product
from collections import defaultdict

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root / "raw/"), train=train, download=True, transform=transform
    )


def addition_dataset(train, num_digits):
    dataset = get_mnist_data(train)
    X, y = dataset.data, dataset.targets
    X = torch.unsqueeze(X, 1).float()
    size = len(X) // num_digits
    X, y = torch.split(X, size), torch.split(y, size)

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    y = torch.sum(torch.stack(y,0),0)
    return X,y



def create_single_digit_addition(num_digits):
    concept_names = ["x%d%d" % (i, j) for i, j in product(range(num_digits), range(10))]


    sums = defaultdict(list)
    for d in product(*[range(10) for _ in range(num_digits)]):
        conj = []
        z= 0
        for i,n in enumerate(d):
            conj.append("x%d%d" % (i,n))
            z += n
        sums[z].append( "(" + " & ".join(conj) + ")")


    explanations= {}
    for z in range(10*num_digits - num_digits + 1):

        explanations["z%d" % z] = {"name": "%d" % z,
                                   "explanation": "(" + " | ".join(sums[z]) + ")"}

    return concept_names, explanations


# MULTI DIGIT
# def create_logic_addition(digit_length):
#     concept_names = ["x%d%d%d" % (i, j, k) for i, j,k in product([1,2], range(digit_length), range(10))]
#
#
#     sums = defaultdict(list)
#
#     for i,j in product(range(10**digit_length), range(10**digit_length)):
#         z = i + j
#         si = str(i)
#         sj = str(j)
#         si = ''.join(["0" for _ in range(digit_length - len(si))]) + si
#         sj = ''.join(["0" for _ in range(digit_length - len(sj))]) + sj
#         xi = ["x1%d%s" %  (digit_length - k - 1,n) for k,n in enumerate(si)]
#         xj = ["x2%d%s" %  (digit_length - k - 1,n) for k,n in enumerate(sj)]
#         conj = "(" + " & ".join(xi + xj) + ")"
#         sums[z].append(conj)
#
#     explanations = {}
#     for z in range(2*(10**digit_length - digit_length) + 1):
#
#         explanations["z%d" % z] = {"name": "z%d" % z,
#                                    "explanation": "(" + " | ".join(sums[z]) + ")"}
#
#     return explanations


if __name__ == '__main__':

    number_digits = 3

    print(create_single_digit_addition(number_digits))

    X, y = addition_dataset(True, 2*number_digits)

    dataset = TensorDataset(*X, y)
    loader = DataLoader(
        dataset,
        batch_size=100
    )

    for batch_idx, I in enumerate(loader):
        X = I[:-1]
        y = I[-1]
        print(batch_idx, y.shape, [x.shape for x in X])

import math
import numpy as np
import sys
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
import torch
from itertools import product
from collections import defaultdict
import random
from pathlib import Path
try:
    from sudoku import Sudoku # pip install py-sudoku
except Exception as e:
    raise Exception("No sudoku package found. Install py-sudoku (e.g. pip install py-sudoku)") from e

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root / "raw/"), train=train, download=True, transform=transform
    )

def sudoku(n):

    m = int(math.sqrt((n)))

    rows = np.reshape(np.arange(0, n * n), [n, n])

    columns = rows.T

    squares = np.array([np.reshape(rows[i:i + m, j:j + m], -1) for i, j in product(range(0, n, m), range(0, n, m))])

    images_per_label = defaultdict(lambda: [])
    dataset = get_mnist_data(True)
    X, y = dataset.data, dataset.targets
    for i in range(len(X)):
        label = int(y[i])
        image = X[i]
        images_per_label[label].append(image)

    labels = []
    images = []
    labels_images = []

    sudokus = list(range(0, 10000, n * n))
    for id_sudoku in sudokus:
        images.append([])
        labels_images.append([])

        seed = random.randint(0, sys.maxsize - 1)
        solution = Sudoku(m,
                          seed=seed).solve().board  # set a fixed seed fro random, so I pass a random one from the outside
        board = np.reshape(solution, [-1]) - 1  # zero-based
        r = random.random()
        if r > 0.5:  # positive
            labels.append([1])
        else:  # negative
            labels.append([0])

            # we corrupt the board #TODO: check that this is a good corruption
            number_corruptions = random.randint(0, n * n)
            t = 0
            while t < number_corruptions:
                a = random.randint(0, n * n - 1)
                b = random.randint(0, n * n - 1)
                if a != b:
                    c = board[a]
                    board[a] = board[b]
                    board[b] = c
                    t += 1
        # print(np.reshape(board, [n,n]))
        # print(labels[-1])

        for i in board:
            images[-1].append(torch.unsqueeze(images_per_label[i].pop(),dim=0))
            labels_images[-1].append(i)
        images[-1] = torch.stack(images[-1], dim=0)


    manifolds = np.concatenate((columns,rows,squares), axis=0)
    tasks = labels

    return images, labels_images, manifolds, tasks



if __name__ == '__main__':
    sudoku(4)

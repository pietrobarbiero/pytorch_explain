import torch
import matplotlib.pyplot as plt

from datasets.sudoku_manifold import sudoku
from model import SudokuRelationalDCR
import numpy as np

import unittest

class SudoluTest(unittest.TestCase):


    def test_sudoku(self):
        n = 4
        X, y, manifolds, task_labels = sudoku(4)

        X = torch.stack(X, dim=0).float()
        y, manifolds, task_labels = (torch.tensor(i) for i in (y, manifolds, task_labels))

        m = SudokuRelationalDCR( manifolds =manifolds, emb_size=10, manifold_arity=n, num_classes=n)

        c, t, _ = m(X)

        return c.shape == torch.eye(n)[y].shape and t.shape == task_labels.shape

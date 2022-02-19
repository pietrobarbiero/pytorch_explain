import torch
from torch.nn.functional import one_hot
import torchvision
import numpy as np


def transform_digit_color(img, color):
    if color == 'red':
        return img * np.array([1, 0, 0])
    elif color == 'green':
        return img * np.array([0, 1, 0])
    elif color == 'blue':
        return img * np.array([0, 0, 1])
    else:
        raise ValueError('Unknown color: ' + color)


def transform_quadrant(img, quadrant):
    if len(img.shape) == 4:
        n, w, h, c = img.shape
        new_imgs = np.zeros((n, 2 * w, 2 * h, c), dtype=np.uint8)
    elif len(img.shape) == 3:
        w, h, c = img.shape
        new_imgs = np.zeros((2 * w, 2 * h, c), dtype=np.uint8)
    else:
        raise ValueError('Invalid img shape: ' + str(img.shape))

    if quadrant == 1:
        new_imgs[..., :w, h:, :] = img
    elif quadrant == 2:
        new_imgs[..., :w, :h, :] = img
    elif quadrant == 3:
        new_imgs[..., w:, :h, :] = img
    elif quadrant == 4:
        new_imgs[..., w:, h:, :] = img
    else:
        raise ValueError('Invalid quadrant: ' + str(quadrant))

    return new_imgs


def transform_background(img, color):
    if color == 'white':
        pass  # img[img == 0] = 0
    elif color == 'black':
        img[img == np.array([0, 0, 0])] = 1
    else:
        raise ValueError('Unknown color: ' + color)
    return img


def select_digits(x, digit_idx, g, examples_per_group):
    if g == -1:
        idxs = np.random.choice(x.shape[0], examples_per_group, replace=True)
    else:
        idxs = np.random.choice(digit_idx[g], examples_per_group, replace=True)
    return idxs
    # selected_xs = x[idxs][..., None].numpy()
    # return selected_xs


def select_colors(selected_xs, g, examples_per_group):
    if g == -1:
        colors = np.random.choice(3, examples_per_group, replace=True)
    else:
        colors = [g] * examples_per_group

    colors = np.eye(3)[colors]
    selected_xs = selected_xs * colors[:, None, None, :]
    return selected_xs


def mnist_poly_task(x, y, digit_idx, groups_per_class, examples_per_group, examples_per_class, aux, xor_task=None, prob_xor=0.5):
    # Choose groups
    groups = np.zeros((2 * groups_per_class, len(aux)), dtype=int) - 1

    # Choose whether XOR task
    if xor_task is None:
        xor_task = np.random.rand() < prob_xor
    if xor_task:
        idxs = np.random.choice(len(aux), len(aux), replace=False)
        idx0 = idxs[0]
        c = np.random.choice(aux[idx0], groups_per_class, replace=False)
        groups[groups_per_class:, idx0] = c
        groups[:groups_per_class, idx0] = c

        idx1 = idxs[1]
        c = np.random.choice(aux[idx1], groups_per_class, replace=False)
        groups[groups_per_class:, idx1] = c
        groups[:groups_per_class, idx1] = c[::-1]
    else:
        idxs = np.random.choice(len(aux), len(aux), replace=False)
        idx0 = idxs[0]
        c1 = np.random.choice(aux[idx0], groups_per_class, replace=True)
        c2 = np.random.choice(list(set(aux[idx0]) - set(c1)), groups_per_class, replace=True)
        groups[groups_per_class:, idx0] = c1
        groups[:groups_per_class, idx0] = c2

    # Append data for each group
    data = []
    digits = []
    colors = []
    for g in groups:
        idxs = [select_digits(x, digit_idx, g[i], examples_per_group) for i in range(4)]
        selected_xs = [x[idx][..., None].numpy() for idx in idxs]
        selected_digits = [y[idx].numpy() for idx in idxs]
        selected_xs = [select_colors(selected_xs[i], g[4 + i], examples_per_group) for i in range(4)]
        selected_colors = [samples.mean(axis=1).mean(axis=1).argmax(axis=1) for samples in selected_xs]

        s = transform_quadrant(selected_xs[0], 1)
        for i in range(1, 4):
            s += transform_quadrant(selected_xs[i], i + 1)

        data.extend(s)
        digits.extend(np.stack(selected_digits).T)
        colors.extend(np.stack(selected_colors).T)

    data = np.stack(data) / 255.
    data = torch.Tensor(data).permute(0, 3, 1, 2)
    digits = np.stack(digits)
    colors = np.stack(colors)
    concepts = np.hstack([digits, colors])
    labels = torch.Tensor([0] * examples_per_class + [1] * examples_per_class)

    digits_1h = np.hstack([one_hot(torch.LongTensor(c), np.max(digits)+1) for c in digits.T])
    colors_1h = np.hstack([one_hot(torch.LongTensor(c), np.max(colors)+1) for c in colors.T])
    concepts_1h = np.hstack([digits_1h, colors_1h])

    return torch.FloatTensor(data), torch.FloatTensor(concepts_1h), torch.FloatTensor(labels)

import itertools

import numpy as np



def get_mean_and_std(x):
    mean = []
    std = []
    for row in x.T.A:
        mean.append(row.mean())
        std.append(row.std())
    return mean, std


def normalize(X, mean, std):
    new_x = []
    for col, m, s in zip(X.T.A, mean, std):
        new_x.append((col - m) / s)
    return np.matrix(new_x).T


def preprocess(X, X_test=None):
    if X_test is not None:
        mean, std = get_mean_and_std(X)
        return normalize(X, mean, std), normalize(X_test, mean, std)
    else:
        mean, std = get_mean_and_std(X)
        return normalize(X, mean, std)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def exclude_row(X, *args):
    return np.delete(X, args, axis=0)


def read_x(path, exclude_y=True, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    _row, col = data.shape
    print('shape {} {}'.format(_row, col))
    sub = 1 if exclude_y else 0
    return np.matrix(exclude_col(data, 0, col - sub)), np.array(data.T[0])  # data, id's


def read_y(path, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)]), np.array(data.T[0])


X, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)


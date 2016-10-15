from random import shuffle

import numpy as np
import numpy.linalg as la


class Regr(object):
    def __init__(self):
        self._fitted = False
        self._beta = None
        self._beta0 = None

    def fit(self, X, y, mean_int=True):
        self._fitted = True
        self._beta = la.inv(X.transpose().dot(X)).dot(X.transpose).dot(y)
        if mean_int:
            self._beta0 = sum(map(lambda a: a[0], y)) / y.row_n
        else:
            self._beta0 = 1

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError('Fit regression before predicting')
        return self._beta.transpose().dot(X) + self._beta


def split(X, y):
    n = y.shape()[0]
    for_test = [i % 2 == 0 for i in range(n)]
    shuffle(for_test)
    X_learn, X_test, y_learn, y_test = [], [], [], []
    for i in range(n):
        if for_test[i]:
            X_test.append(X[i])
            y_test.append(y[0])
        else:
            X_learn.append(X[i])
            y_learn.append(y[0])
    return np.matrix(X_learn), np.matrix(X_test), np.array(y_learn), np.array(y_test)


def rmse(y, y_est):
    return np.sqrt(((y - y_est) ** 2).mean())


def scale(X):
    scaled = []
    for i in range(X.shape[1]):
        col = X[:, i]
        mean = col.mean()
        std = col.std()
        scaled.append([(col[i, 0] - mean) / std for i in range(len(col))])
    return np.matrix(scaled).transpose()


def read(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    return exclude_col(data, 0)  # get rid of id's


def write(path, data):
    np.savetxt(path, data, header='id,target', delimiter=',', comments='')


def exclude_row(X, *args):
    return np.delete(X, args, axis=0)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)
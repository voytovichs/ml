from random import shuffle, random

import numpy as np
import numpy.linalg as la


class Regr(object):
    def __init__(self):
        self._fitted = False
        self._beta = None
        self._beta0 = None

    def fit(self, X, y, mean_int=True):
        self._fitted = True
        self._beta = (y - y.mean()) * (la.inv(X * X.transpose()).dot(X))
        if mean_int:
            self._beta0 = y.mean()
        else:
            self._beta0 = 1

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError('Fit regression before predicting')
        y = self._beta * X.T + self._beta0
        arr = [y[0, i] for i in range(y.shape[1])]
        return np.array(arr)

def split(X, y, seventy_five=False):
    row, col = X.shape
    for_test = [i % 2 == 0 for i in range(row)]
    if seventy_five:
        # learn 75 / test 25
        for_test = list(map(lambda a: random.choice(0, 1) if a else a, for_test))
    shuffle(for_test)
    X_learn, X_test, y_learn, y_test = [], [], [], []
    for i in range(row):
        if for_test[i]:
            X_test.append([X[i, k] for k in range(col)])
            y_test.append(y[i])
        else:
            X_learn.append([X[i, k] for k in range(col)])
            y_learn.append(y[i])
    return np.matrix(X_learn), np.matrix(X_test), np.array(y_learn), np.array(y_test)


def split_rep(X, y):
    pass


def split_k_fold(X, y, k):
    n = len(y)
    variance = [i % k for i in range(n)]
    folds_x = [[] for i in range(k)]
    folds_y = [[] for i in range(k)]
    for i in range(len(variance)):
        folds_x[variance[i]].append(X[i])
        folds_y[variance[i]].append(y[i])
    return folds_x, folds_y


def rmse(y, y_est):
    return np.sqrt(np.mean(((y - y_est) ** 2)))


def scale(X):
    scaled = []
    _, col = X.shape
    for i in range(col):
        col = X[:, i]
        col_len = col.shape[0]
        mean = col.mean()
        std = col.std()
        scaled.append([(col[k] - mean) / std for k in range(col_len)])
    return np.matrix(scaled).transpose()


def read_x(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    _row, col = data.shape
    return exclude_col(data, 0, col - 1)  # get rid of id's and y


def read_y(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)])


def write(path, data):
    np.savetxt(path, data, header='id,target', delimiter=',', comments='')


def exclude_row(X, *args):
    return np.delete(X, args, axis=0)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def cv(X, y, times=1000):
    err = np.zeros(times)
    X_sc = scale(X)
    X_l, X_t, y_l, y_t = split(X_sc, y)
    for i in range(times):
        r = Regr()
        r.fit(X_l, y_l)
        y_est = r.predict(X_t)
        err[i] = rmse(y_t, y_est)
    return err.mean()


X = read_x('learn.csv')
y = read_y('learn.csv')
print(cv(X, y))

from math import sqrt
from random import shuffle

from matrix import Matrix


class LinRerg(object):
    def __init__(self):
        self._fitted = False
        self._beta = None
        self._beta0 = None

    def fit(self, X, y):
        self._fitted = True
        self._beta = (X.get_transposed() * X).get_inverted() * X.get_transposed() * y
        self._beta0 = sum(map(lambda a: a[0], self._beta)) / self._beta.shape()[0]

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError('Fit regression before predicting')
        return (self._beta.get_transposed() * X)[0][0] + self._beta0


def read(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            num_line = list(map(float, line.split(',')))
            n = len(num_line)
            X.append(num_line[0: n - 1])
            y.append(num_line[n - 1])
    return Matrix(X), Matrix(y)


def rmse(y, y_est):
    return sum([sqrt(abs(y[i] - y_est[i])) for i in range(len(y))]) / len(y)


def split(X, y):
    n = y.shape()[0]
    for_test = [i % 2 == 0 for i in range(n)]
    shuffle(for_test)
    X_learn, X_test, y_learn, y_test = [], [], [], []
    for i in range(n):
        if for_test[i]:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_learn.append(X[i])
            y_learn.append(y[i])
    return Matrix(X_learn), Matrix(X_test), Matrix(y_learn), Matrix(y_test)

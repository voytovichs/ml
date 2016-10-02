from math import sqrt
from random import shuffle

from matrix import Matrix


class LinRegr(object):
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
        ha = (self._beta.get_transposed() * Matrix(X))
        return ha[0][0] + self._beta0


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
    n = y.row_n
    return sum([sqrt(abs(y[i][0] - y_est[i])) for i in range(n)]) / n


def split(X, y):
    n = y.shape()[0]
    for_test = [i % 2 == 0 for i in range(n)]
    shuffle(for_test)
    X_learn, X_test, y_learn, y_test = [], [], [], []
    for i in range(n):
        if for_test[i]:
            X_test.append(X[i])
            y_test.append(y[i][0])
        else:
            X_learn.append(X[i])
            y_learn.append(y[i][0])
    return Matrix(X_learn), Matrix(X_test), Matrix(y_learn), Matrix(y_test)


def cross_validation(X, y, repeat=1000):
    X_learn, X_Test, y_learn, y_test = split(X, y)
    error = 0
    for i in range(repeat):
        try:
            regression = LinRegr()
            regression.fit(X_learn, y_learn)
            y_est = [regression.predict(X_Test[i]) for i in range(X_Test.row_n)]
            error += rmse(y_test, y_est) / repeat
        except:
            # singular matrix, ignore iteration
            pass
    return error


X, y = read('/Users/voytovichs/Code/ml/students.txt')
err = cross_validation(X, y)

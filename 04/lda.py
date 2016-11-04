import itertools

import numpy as np
import sklearn
from random import shuffle


class LDA:
    def __init__(self):
        self._coef = None

    def _mean_vec(self, x):
        means = []
        for column in x.T:
            means.append(column.mean())
        return np.array(means)

    def _split_by_classes(self, X, y):
        x1 = []
        x2 = []
        iterable_x = X.A
        for i in range(len(y)):
            if y[i] == 0:
                x1.append(iterable_x[i])
            else:
                x2.append(iterable_x[i])
        return np.matrix(x1), np.matrix(x2)

    def _cov_matrix(self, xs):
        covs = [np.cov(x.T, bias=1) for x in xs]
        return np.average(covs, axis=0, weights=list(map(lambda x: len(x), xs)))

    def fit(self, X, y):
        x1, x2 = self._split_by_classes(X, y)
        x1_mean = self._mean_vec(x1)
        x2_mean = self._mean_vec(x2)
        cov = self._cov_matrix([x1, x2])
        den = [len(x1), len(x2)]
        self.coef = np.dot(np.linalg.inv(cov), (x1_mean - x2_mean).T)
        self.coef -= np.dot(np.dot(1 / float(2) * (x1_mean - x2_mean).T, np.linalg.inv(cov)),
                            (x1_mean - x2_mean))
        self.coef += np.log(den[0] / float(den[1]))

    def predict(self, X):
        result = np.dot(X, self.coef).A[0]
        return (result > 0).astype(np.int)


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


def auc(y, y_predicted):
    return sklearn.metrics.roc_auc_score(y, y_predicted)


def split(X, y):
    row, col = X.shape
    for_test = [i % 2 == 0 for i in range(row)]
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


def cv(X, y, model, n=100):
    aucs = []
    for i in range(n):
        X_l, X_t, y_l, y_t = split(X, y)
        r = model()
        r.fit(X_l, y_l)
        y_est = r.predict(X_t)
        aucs.append(auc(y_t, y_est))
    return np.median(aucs)


X, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)
X, test = preprocess(X, test)
a = LDA()
a.fit(X, y)
est = a.predict(X)
print(auc(y, est))
print(cv(X, y, LDA))

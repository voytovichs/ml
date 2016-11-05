import itertools
import os

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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
        n = sum([len(xs[0]), len(xs[1])])
        return np.average(covs, axis=0, weights=[len(xs[0]) / float(n), len(xs[1]) / float(n)])

    def _regr(self, x, y):
        first = x.dot(x.T)
        second = np.linalg.inv(first)
        third = second.dot(x)
        fourth = (y - y.mean()).T
        return fourth.dot(third).T

    def fit(self, X, y):
        X = X.A
        x1, x2 = X[y == 0, :], X[y == 1, :]
        means = np.asarray([x1.mean(0), x2.mean(0)])
        cov = self._cov_matrix([x1, x2])
        n = len(X)
        den = np.array([len(x1) / float(n), len(x2) / float(n)])
        self.w = self._regr(cov, means.T).T
        self.int = (-0.5 * np.diag(np.dot(means, self.w.T)) + np.log(den))
        self.w = np.array(self.w[1, :] - self.w[0, :], ndmin=2)
        self.int = np.array(self.int[1] - self.int[0], ndmin=1)

    def predict(self, X):
        result = np.array((np.dot(X, self.w.T) + self.int)).ravel()
        result -= result.mean()
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


def write(path, data, ids):
    tmp = 'haha.csv'
    np.savetxt(tmp, data, fmt='%d', header='id,label', delimiter=',', comments='')
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as f:
        f.writelines(lines)


def conc_column(a, b):
    return np.hstack((a, b))


def concatenate_matrices(a, b):
    return np.vstack((a, b))


def multiply_features(X, dim=3):
    dims = [np.matrix.copy(X.T)]
    for d in range(1, dim):
        new_dimension = []
        n = dims[d - 1].shape[0]
        for i1 in range(n):
            for i2 in range(i1, n):
                a1 = dims[d - 1][i1]
                a2 = dims[d - 1][i2]
                new_dimension.append(np.squeeze(np.asarray(np.multiply(a1, a2))))
        dims.append(np.matrix(new_dimension))
    result = dims[0].T
    for i in range(1, len(dims)):
        result = conc_column(result, dims[i].T)
    return result


def split_matrix(a, bound=336):
    return a[:bound], a[bound:]


def get_rid_of_outliers(x, y, mean, std, m=3):
    to_delete = set()
    row, col = x.shape
    for c in range(col):
        for r in range(row):
            if abs(x[r, c] - mean[c]) >= m * std[c]:
                to_delete.add(r)
    print('{0} rows have outline values'.format(len(to_delete)))
    return exclude_row(x, *to_delete), exclude_row(y, *to_delete)


X, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)
X, test = preprocess(X, test)
X, test = split_matrix(multiply_features(concatenate_matrices(X, test), dim=2), len(X))
m, s = get_mean_and_std(X)
X, y = get_rid_of_outliers(X, y, m, s, m=15)

a = LDA()
a.fit(X, y)
est = a.predict(X)
y_test = a.predict(test)
write('answer.csv', y_test, test_id)
print(auc(y, est))
print(cv(X, y, LDA))

import os
from collections import OrderedDict
import sys
import subprocess
import numpy as np


# TODO: read http://www.machinelearning.ru/wiki/index.php?title=KNN

class KNN:
    def __init__(self, X, y):
        self._X = X.A
        self._y = y
        self._n = len(y)

    def _euclid_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _distance(self, x, y):
        return self._euclid_distance(x, y)

    def fit(self, X, one_left_out=None):
        self._neigh = []
        for a in X.A:
            neighbours = []
            for i in range(self._n):
                # format is (distance, label)
                if one_left_out == i:
                    continue
                neighbours.append((self._distance(a, self._X[i]), self._y[i]))
            s = sorted(neighbours, key=lambda _p: _p[0])
            self._neigh.append(s)

    def _compare_pair(self, a, b):
        if a[0] < b[0]:
            return - 1
        elif a[0] > b[0]:
            return 1
        else:
            return 0

    def _make_decision(self, neighbours):
        dict = OrderedDict()
        for a in neighbours:
            if a[1] in dict:
                dict[a[1]] += 1
            else:
                dict[a[1]] = 1
        s = sorted(dict.items(), key=lambda _p: _p[1], reverse=True)
        return s[0][0]

    def predict(self, X, k):
        if self._neigh is None:
            raise Exception('Call fit first')
        labels = []
        for i in range(len(X)):
            label = self._make_decision(self._neigh[i][:k])
            labels.append(label)
        return np.array(labels)


def read_x(path, exclude_y=True, n=None):
    data = np.genfromtxt(path, delimiter=',', skip_header=True, max_rows=n)
    _row, col = data.shape
    sub = 1 if exclude_y else 0
    return np.matrix(exclude_col(data, 0, col - sub)), np.array(data.T[0])  # data, id's


def read_y(path, n=None):
    data = np.genfromtxt(path, delimiter=',', skip_header=True, max_rows=n)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)]), np.array(data.T[0])


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def accuracy(y, y_real):
    n = len(y)
    cnt = [1 if y[i] == y_real[i] else 0 for i in range(n)]
    return sum(cnt) / n


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def find_zero_columns(X):
    zeros = []
    i = -1
    for col in X.T.A:
        i += 1
        if np.all(col == 0):
            zeros.append(i)
    return zeros


def delete_zero_columns(X, X_test=None):
    zeros = find_zero_columns(X)
    if X_test != None:
        return exclude_col(X, *zeros), exclude_col(X_test, *zeros)
    return exclude_col(X, *zeros)


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
    if X_test != None:
        nx, nx_test = delete_zero_columns(X, X_test)
        mean, std = get_mean_and_std(nx)
        return normalize(nx, mean, std), normalize(nx_test, mean, std)
    else:
        nx = delete_zero_columns(X)
        mean, std = get_mean_and_std(nx)
        return normalize(nx, mean, std)


# TODO: split on three parts: learn, test_k, test
def cv(X, y, k=(3, 10, 20, 40), log=True):
    n = len(y)
    labels = list(map(lambda a: [], [None] * len(k)))
    for i in range(n):
        for k_ind in range(len(k)):
            knn = KNN(X, y)
            knn.fit(X[i], one_left_out=i)
            label = knn.predict(X[i], k[k_ind])
            labels[k_ind].append(label[0])
        if log:
            print('Object {} classified'.format(i))

    acc = [accuracy(labels[i], y) for i in range(len(labels))]
    return acc


def write(path, data, ids):
    tmp = 'haha.csv'
    np.savetxt(tmp, data, fmt='%d', header='id,label', delimiter=',', comments='')
    lines = []
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as f:
        f.writelines(lines)


if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

X, x_id = read_x('learn.csv', n=1000)
y, y_id = read_y('learn.csv', n=1000)
X = preprocess(X)

print(cv(X, y))
'''
if __name__ == '__main__':
    X, x_id = read_x('learn.csv')
    y, y_id = read_y('learn.csv')
    test, test_id = read_x('learn.csv')
    knn = KNN(X, y)
    knn.fit(test)
    labels = knn.predict(X, 20)  # TODO: fix me!
    write('answer.csv', labels, test_id)
'''

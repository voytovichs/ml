import os
from collections import OrderedDict
from random import shuffle

import numpy as np


#TODO: read http://www.machinelearning.ru/wiki/index.php?title=KNN

class KNN:
    def __init__(self):
        self.fitted = False

    def _euclid_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _distance(self, x, y):
        return self._euclid_distance(x, y)

    def fit(self, X, y):
        self._X = X.A
        self._y = y
        self._n = len(y)

    # may not work because of self argument
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
        s = sorted(dict.iteritems(), key=lambda _p: _p[1], reverse=True)
        return s[0][0]


    def prepare_to_predict(self, X):
        self._neigh = []
        for a in X.A:
            neighbours = []
            for i in range(self._n):
                # format is (distance, label)
                neighbours.append((self._distance(a, self._X[i]), self._y[i]))
            s = sorted(neighbours, key=lambda _p: _p[0])
            self._neigh.append(s)


    def predict(self, X, k):
        if self._neigh is None:
            raise Exception('Call prepare_to_predict first')
        labels = []
        for i in range(len(X.A)):
            i += 1
            label = self._make_decision(self._neigh[i][:k])
            labels.append(label)
        return np.array(labels)


def read_x(path, exclude_y=True):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    _row, col = data.shape
    sub = 1 if exclude_y else 0
    return np.matrix(exclude_col(data, 0, col - sub))  # get rid of id's and y


def read_y(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)])


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)

# TODO: this doesn't work, id's aren't there
def write(path, data):
    tmp = 'haha.csv'
    ids = [i + 336 for i in range(len(data))]
    np.savetxt(tmp, data, fmt='%d', header='id,label', delimiter=',', comments='')
    lines = []
    with open(tmp, 'r') as file:
        lines = file.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0},{1}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as file:
        file.writelines(lines)


def accuracy(y, y_real):
    n = len(y)
    cnt = [1 if y[i] == y_real[i] else 0 for i in range(n)]
    return sum(cnt) / n


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


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)

def find_zero_columns(X):
    zeros = []
    i = -1
    for col in X.T.A:
        i+=1
        if np.all(col==0):
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

# TODO: implement leave-one-out
# TODO: split on three parts: learn, test_k, test

def cv(X, y, iterations=10, log=False):
    acc = np.zeros(iterations)
    for i in range(iterations):
        X_l, X_t, y_l, y_t = split(X, y)
        knn = KNN()
        print('Call fit')
        knn.fit(X_l, y_l)
        print('Call predict')
        knn.prepare_to_predict(X_t)
        y_est = knn.predict(X_t, 5)
        acc[i] = accuracy(y_est, y_t)
        print('Accurancy: {}'.format(acc[i]))
    if log:
        print(acc)
    return np.median(acc)

X, y = preprocess(read_x('fake_learn.fake')), read_y('fake_learn.fake')

print(cv(X, y, 10))

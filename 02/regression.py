import os
from random import shuffle

import numpy as np
import numpy.linalg as la

class Regr(object):
    def __init__(self):
        self._fitted = False
        self._beta = None
        self._beta0 = None

    def fit(self, X, y):
        self._fitted = True
        ha = X.dot(X.T)
        #self._beta = (y - y.mean()).dot((la.inv(X.dot(X.T)).dot(X)))
        self._beta = la.inv(X * X.T)
        self._beta0 = y.mean()

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError('Fit regression before predicting')
        y = self._beta.T * X + self._beta0
        arr = [y[0, i] for i in range(y.shape[1])]
        return np.array(arr)


def split(X, y, seventy_five=False):
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


def split_array(X, y):
    row = len(X)
    for_test = [i % 2 == 0 for i in range(row)]
    shuffle(for_test)
    X_learn, X_test, y_learn, y_test = [], [], [], []
    for i in range(row):
        if for_test[i]:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_learn.append(X[i])
            y_learn.append(y[i])
    return np.array(X_learn), np.array(X_test), np.array(y_learn), np.array(y_test)


def rmse(y, y_est):
    return np.sqrt(np.mean(((y - y_est) ** 2)))


def scale(X, mean, std):
    scaled = []
    _, col = X.shape
    for i in range(col):
        col = X[:, i]
        col_len = col.shape[0]
        scaled.append([(col[k] - mean[i]) / std[i] for k in range(col_len)])
    return np.matrix(scaled).T


def read_x(path, exclude_y=True):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    _row, col = data.shape
    sub = 1 if exclude_y else 0
    return exclude_col(data, 0, col - sub)  # get rid of id's and y


def read_y(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)])


def write(path, data):
    tmp = 'haha.csv'
    ids = [i + 336 for i in range(len(data))]
    np.savetxt(tmp, data, fmt='%.15f', header='id,target', delimiter=',', comments='')
    lines = []
    with open(tmp, 'r') as file:
        lines = file.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0},{1}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as file:
        file.writelines(lines)


def exclude_row(X, *args):
    return np.delete(X, args, axis=0)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def cv(X, y, array=False, iterations=1000, log=False):
    err = np.zeros(iterations)
    for i in range(iterations):
        X_l, X_t, y_l, y_t = split(X, y) if not array else split_array(X, y)
        r = Regr()
        if array:
            X_l = X_l.reshape(-1, 1)
            X_t = X_t.reshape(-1, 1)
        r.fit(X_l, y_l)

        y_est = r.predict(X_t if not array else np.matrix(X_t))
        err[i] = rmse(y_t, y_est)
    if log:
        print(err)
    return np.median(err)


def get_mean_and_std(x):
    mean = []
    std = []
    for row in x.transpose():
        mean.append(row.mean())
        std.append(row.std())
    return mean, std


def get_rid_of_outliers(x, y, mean, std, m=3):
    to_delete = set()
    row, col = x.shape
    for c in range(col):
        for r in range(row):
            if abs(x[r, c] - mean[c]) >= m * std[c]:
                to_delete.add(r)
    print('{0} rows has outline values'.format(len(to_delete)))
    return exclude_row(x, *to_delete), exclude_row(y, *to_delete)


def correlation(a, b):
    a_mean = a.mean()
    b_mean = b.mean()
    n = len(b)
    cov = sum([(a[0, i] - a_mean) * (b[i] - b_mean) for i in range(n)])
    return cov / (a.std() * b.std() * n)


def add_column(X, col):
    return np.vstack((X.T, col)).T


def backward_selection(X, y, min_col=1, cv_it=500):
    to_delete = set()
    _, n = X.shape
    err = float('inf')
    new_err = err - 1
    while n - len(to_delete) > min_col:
        err = new_err
        new_err = float('inf')
        ind = -1
        min_ind = -1
        for col in X.T:
            ind += 1
            if ind in to_delete:
                continue
            cur_err = cv(exclude_col(X, ind, *to_delete), y, iterations=cv_it)
            if cur_err < new_err:
                new_err = cur_err
                min_ind = ind
        if min_ind != -1:
            to_delete.add(min_ind)
        print('Error on new iteration {}, get rid of {}'.format(new_err, min_ind))
    print(to_delete)
    return to_delete


def recover_index(a):
    recovered = a[:]
    for i in range(len(recovered)):
        for k in range(i + 1, len(recovered)):
            if recovered[k] > recovered[i]:
                recovered[k] += 1
    return recovered


def exclude_col_except(X, *exc):
    row, col = X.shape
    to_delete = set(i for i in range(col)).difference(exc)
    return exclude_col(X, *to_delete)


def select_one_by_one(X, y, cv_iter, n):
    selected = []
    learn_set = exclude_col(X)  # weird kind of copying
    for iter in range(n):
        min_err = float('inf')
        min_ind = -1
        ind = min_ind
        for col in learn_set.T:
            ind += 1
            cur_err = cv(col, y, array=True, iterations=cv_iter)
            if cur_err < min_err:
                min_err = cur_err
                min_ind = ind
        learn_set = exclude_col(learn_set, min_ind)
        selected.append(min_ind)
        print('Current error {}'.format(min_err))
    return recover_index(selected)




X = read_x('learn.csv')
y = read_y('learn.csv')
learn_mean, learn_std = get_mean_and_std(X)
X, y = get_rid_of_outliers(X, y, learn_mean, learn_std, m=10)
selected = set(select_one_by_one(X, y, cv_iter=50, n=30))
X = exclude_col_except(X, *selected)

test = read_x('test.csv', exclude_y=False)
test = exclude_col_except(test, *selected)

r = Regr()
r.fit(X, y)
#print(r._beta)

answer = r.predict(test)
print(answer)
write('answer.csv', answer)
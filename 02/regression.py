import os
from random import shuffle, random, choice

import numpy as np
import numpy.linalg as la


class Regr(object):
    def __init__(self):
        self._fitted = False
        self._beta = None
        self._beta0 = None

    def fit(self, X, y, mean_int=True):
        self._fitted = True
        self._beta = (y - y.mean()).dot((la.inv(X.dot(X.transpose())).dot(X)))
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


def scale(X, mean, std):
    scaled = []
    _, col = X.shape
    for i in range(col):
        col = X[:, i]
        col_len = col.shape[0]
        scaled.append([(col[k] - mean[i]) / std[i] for k in range(col_len)])
    return np.matrix(scaled).transpose()


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


def cv(X, y, times=1000):
    err = np.zeros(times)
    X_l, X_t, y_l, y_t = split(X, y)
    for i in range(times):
        r = Regr()
        r.fit(X_l, y_l)
        y_est = r.predict(X_t)
        err[i] = rmse(y_t, y_est)
    return err.mean()


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


def max_corr_index(X, y):
    best = 0
    second_i = -1
    third_i = -1
    i = -1
    curr = 0
    for col in X.T:
        corr = correlation(col, y)
        if corr > best:
            third_i = second_i
            second_i = i
            best = corr
            i = curr
        curr += 1
    return i, second_i, third_i


def exclude_col_except(X, *exc):
    row, col = X.shape
    to_delete = set(i for i in range(col)).difference(exc)
    return exclude_col(X, *to_delete)


def add_column(X, col):
    return np.vstack((X.T, col)).T


def forward_selection(X, y, fine=0, times=500):
    best, second, third = max_corr_index(X, y)
    selected = set()
    selected.add(best)
    selected.add(second)
    selected.add(third)
    X_to_learn = exclude_col_except(X, best, second, third)
    err = 100000000000000
    new_err = cv(X_to_learn, y, times=times)
    while new_err < err or fine > 0:
        err = new_err
        ind = -1
        min_ind = -1
        for col in X.T:
            ind += 1
            if ind in selected:
                continue
            cur_err = cv(add_column(X_to_learn, col), y, times=times)
            if cur_err < new_err:
                new_err = cur_err
                min_ind = ind
        if min_ind != -1:
            selected.add(min_ind)
            X_to_learn = add_column(X_to_learn, X.T[min_ind])
        elif fine > 0:
            fine -= 1
            rand = choice(list(set(range(X.shape[1])).difference(selected)))
            selected.add(rand)
            X_to_learn = add_column(X_to_learn, X.T[rand])
    return selected


X = read_x('learn.csv')
y = read_y('learn.csv')
X_test = read_x('test.csv', exclude_y=False)
learn_mean, learn_std = get_mean_and_std(X)
X_learn_scaled = scale(X, learn_mean, learn_std)
keep = forward_selection(X_learn_scaled, y, fine=50, times=500)
X_learn_scaled_col = exclude_col_except(X, *keep)
X_test_col = exclude_col_except(X_test, *keep)

X_learn_scaled_col_row, y_row = get_rid_of_outliers(X_learn_scaled_col, y, learn_mean, learn_std, m=25)
X_test_col_scaled = scale(X_test_col, learn_mean, learn_std)
r = Regr()
r.fit(X_learn_scaled_col_row, y_row)
y_test = r.predict(X_test_col_scaled)
write('answer.csv', y_test)
cv(X_learn_scaled_col_row, y_row)

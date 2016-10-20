import os
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
        for_test = list(map(lambda a: np.random.choice([0, 1]) if a else a, for_test))
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


def cv(X, y, iterations=1000, log=False):
    err = np.zeros(iterations)
    X_l, X_t, y_l, y_t = split(X, y, seventy_five=False)
    for i in range(iterations):
        r = Regr()
        r.fit(X_l, y_l)
        y_est = r.predict(X_t)
        err[i] = rmse(y_t, y_est)
    if log:
        print(err)
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


def forward_selection(X, y, fine=0, min_col=100, cv_it=500):
    best, second, third = max_corr_index(X, y)
    selected = set()
    selected.add(best)
    selected.add(second)
    selected.add(third)
    X_to_learn = exclude_col_except(X, best, second, third)
    err = 100000000000000
    new_err = cv(X_to_learn, y, iterations=cv_it)
    while new_err < err or err > 3 and len(selected) < min_col:
        err = new_err
        new_err = 100000000000000
        ind = -1
        min_ind = -1
        for col in X.T:
            ind += 1
            if ind in selected:
                continue
            cur_err = cv(add_column(X_to_learn, col), y, iterations=cv_it, log=False)
            if cur_err < new_err:
                new_err = cur_err
                min_ind = ind
        if min_ind != -1:
            selected.add(min_ind)
            X_to_learn = add_column(X_to_learn, X.T[min_ind])
        print('Error on new iteration {}'.format(err))
    print('Select {0} columns'.format(len(selected)))
    print(selected)
    return selected



X = read_x('learn.csv')
y = read_y('learn.csv')
learn_mean, learn_std = get_mean_and_std(X)

X_learn_scaled = scale(X, learn_mean, learn_std)
X_learn_scaled_row, y_row = get_rid_of_outliers(X_learn_scaled, y, learn_mean, learn_std, m=15)

X_test = read_x('test.csv', exclude_y=False)
X_test_scaled = scale(X_test, learn_mean, learn_std)

keep = [0, 1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 17, 19, 20, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 61, 62, 63, 64, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 114, 116, 117, 118, 119, 120, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 141, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 166, 167, 168, 169, 170, 173, 174, 176, 178, 179, 180, 181, 182, 183, 185, 186, 187, 189, 190, 191, 192, 194, 195, 196, 197, 198]
X_learn_scaled_row_col = exclude_col_except(X_learn_scaled_row, *keep)
X_test_col = exclude_col_except(X_test_scaled, *keep)
print(cv(X_learn_scaled_row_col, y_row, iterations=500))
r = Regr()
r.fit(X_learn_scaled_row_col, y_row)
y_test = r.predict(X_test_col)
write('answer.csv', y_test)


'''
learn_mean, learn_std = get_mean_and_std(X)
X_learn_scaled = scale(X, learn_mean, learn_std)
X_learn_scaled_row, y_row = get_rid_of_outliers(X_learn_scaled, y, learn_mean, learn_std, m=25)

keep = forward_selection(X_learn_scaled_row, y_row, min_col=170, cv_it=5)
X_learn_scaled_row_col = exclude_col_except(X_learn_scaled_row, *keep)

X_test = read_x('test.csv', exclude_y=False)
X_test_col_scaled = scale(X_test_col, learn_mean, learn_std)
X_test_col = exclude_col_except(X_test, *keep)

r = Regr()
r.fit(X_learn_scaled_row_col, y_row)
y_test = r.predict(X_test_col_scaled)
write('answer.csv', y_test)
print(cv(X_learn_scaled_row_col, y_row, iterations=500))

'''
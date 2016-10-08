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
        self._beta0 = sum(map(lambda a: a[0], y)) / y.row_n

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


def mean(vec):
    return sum(vec) / len(vec)


def std(vec, mean=None):
    if not mean:
        mean = mean(vec)
    return sqrt(sum([(vec[i] - mean) ** 2 for i in range(len(vec))]) / len(vec))


def rmse(y, y_est):
    n = y.row_n
    return sqrt(sum([((y[i][0] - y_est[i]) ** 2) for i in range(n)]) / n)


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
    err = 0
    skipped = 0
    succeeded = 0
    for i in range(repeat):
        X_learn, X_Test, y_learn, y_test = split(X, y)
        try:
            regression = LinRegr()
            regression.fit(X_learn, y_learn)
            y_est = [regression.predict(X_Test[k]) for k in range(X_Test.row_n)]
            new_err = rmse(y_test, y_est) / repeat
            if new_err > 50:
                print('Huge error {0}, skip iteration'.format(new_err))
            else:
                err += new_err
            succeeded += 1
        except Exception:
            # singular matrix, ignore iteration
            skipped += 1
            continue
    print('Skipped {0} iterations, succedded {1}'.format(skipped, succeeded))
    return err


def scale(X, skip_binary=False):
    for i in range(X.col_n):
        col = [X[k][i] for k in range(X.row_n)]
        if skip_binary:
            for a in col:
                if a != 0 and a != 1:
                    continue
        m_max = max(col)
        m_min = min(col)
        for j in range(X.row_n):
            X[j][i] = (X[j][i] - m_min) / (m_max - m_min)


def scale_y(y):
    m = mean(list(map(lambda a: a[0], y.rows)))
    for i in range(y.row_n):
        y[i][0] -= m


X, y = read('/Users/voytovichs/Code/ml/students.txt')
scale(X, skip_binary=True)
scale_y(y)
prev_err = cross_validation(X, y)
print('Result with all columns - {0}'.format(prev_err))
while False:
    error = []
    for i in range(X.col_n):
        error.append(cross_validation(X.get_with_exluded_column(i), y))
    _min = error[0]
    min_index = 0
    for i in range(len(error)):
        if _min > error[i]:
            _min = error[i]
            min_index = i
    print('Excluding column {0} gives {1} error'.format(min_index, _min))
    if _min < prev_err:
        prev_err = _min
        X = X.get_with_exluded_column(min_index)
        print('Continue experiment')
    else:
        print('New error without {0} column was {1} which is greater than old {2}. Stopping.'
              .format(min_index, _min, prev_err))
        break
# Excluding column 7 gives 4.36924600085931 error
# Excluding column 9 gives 3.365243424196011 error
# Excluding column 7 gives 2.8821635633293834 error
# Excluding column 0 gives 2.4088621131679666 error
# Excluding column 7 gives 2.1602404740368053 error
# Excluding column 6 gives 2.0034290996177577 error
# Excluding column 3 gives 1.8583080014552404 error
# Excluding column 0 gives 1.7879822077737186 error
# Excluding column 1 gives 1.676595284400759 error
exclude = [7, 9, 7, 0, 7, 6, 3, 0, 1]
X = X.get_with_excluded_columns(exclude)
err = cross_validation(X, y)
print('Final result = {0}'.format(err))

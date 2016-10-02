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

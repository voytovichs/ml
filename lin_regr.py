from matrix import Matrix


class LinRerg(object):
    def fit(self, X, y):
        x_x = X.get_transposed() * X
        print(x_x.get_inverted())
        return list(map(lambda a: a[0], (X.get_transposed() * X).get_inverted() * X.get_transposed() * y))

    def read(self, file_path):
        X = []
        y = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.replace('\n', '')
                num_line = list(map(float, line.split(',')))
                n = len(num_line)
                X.append(num_line[0: n - 1])
                y.append(num_line[n - 1])
        for i in range(len(X)):
            X[i].append(1)
        return Matrix(X), Matrix(y)

def fit(X, y):
    return X * X.get


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
    return X, y

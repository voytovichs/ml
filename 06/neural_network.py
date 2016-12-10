import itertools
import os
import random

import numpy as np


class NeuralNetwork:
    def __init__(self, layers, learning_rate, epochs):
        self.fitted_ = False
        self.biases_ = [np.random.randn(a, 1) for a in layers[1:]]
        self.weights_ = [np.random.randn(a, b) for b, a in zip(layers[:-1], layers[1:])]

        self.num_layers_ = len(layers)
        self.layers_ = layers
        self.learning_rate_ = learning_rate
        self.epochs_ = epochs

        print('Initialized Network:')
        print(' Layers={}'.format(self.layers_))
        print(' LearningRate={}'.format(self.learning_rate_))
        print(' Epochs={}'.format(self.epochs_))

    def backpropagation_(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases_]
        nabla_w = [np.zeros(w.shape) for w in self.weights_]
        activations = [np.matrix(x).T.A]
        zs = []
        for b, w in zip(self.biases_, self.weights_):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(zs[-1]))
        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers_):
            delta = np.dot(self.weights_[-l + 1].transpose(), delta) * sigmoid_derivative(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def fit(self, x, y):
        for epoch in range(self.epochs_):
            c_b = [np.zeros(b.shape) for b in self.biases_]
            c_w = [np.zeros(w.shape) for w in self.weights_]
            for sample, answer in zip(x, y):
                delta_c_b, delta_c_w = self.backpropagation_(np.array(sample), answer)
                c_b = [nb + dnb for nb, dnb in zip(c_b, delta_c_b)]
                c_w = [nw + dnw for nw, dnw in zip(c_w, delta_c_w)]
            self.weights_ = [w - (self.learning_rate_ / len(x)) * nw for w, nw in zip(self.weights_, c_w)]
            self.biases_ = [b - (self.learning_rate_ / len(x)) * nb for b, nb in zip(self.biases_, c_b)]
            print('Finished {0}'.format(epoch))
        self.fitted_ = True

    def predict(self, test_x):
        if not self.fitted_:
            raise Exception('Call fit first!')
        result = []
        for x in test_x:
            a = np.matrix(x).T.A
            for b, w in zip(self.biases_, self.weights_):
                a = sigmoid(np.dot(w, a) + b)
            result.append(a[0, 0])
        return result


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def exclude_col(m, *columns):
    return np.delete(m, columns, axis=1)


def exclude_row(m, *columns):
    return np.delete(m, columns, axis=0)


def read_x(path, exclude_y=True, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    _row, col = data.shape
    sub = 1 if exclude_y else 0
    return np.matrix(exclude_col(data, 0, col - sub)).A, np.array(data.T[0])


def normalize(x, mean, std):
    normalized = [(col - m) / s for col, m, s in zip(x.T, mean, std)]
    return np.matrix(normalized).T.A


def get_mean_and_std(x):
    mean = [row.mean() for row in x.T]
    std = [row.std() for row in x.T]
    return mean, std


def read_y(path, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)]), np.array(data.T[0])


def preprocess(x, x_test=None):
    if x_test is not None:
        mean, std = get_mean_and_std(x)
        return normalize(x, mean, std), normalize(x_test, mean, std)
    else:
        mean, std = get_mean_and_std(x)
        return normalize(x, mean, std)


def write_answer(path, data, ids):
    tmp = '~tmp.trash'
    np.savetxt(tmp, data, fmt='%f', header='id,label', delimiter=',', comments='')
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1:f}\n'.format(ids[i - 1], float(lines[i]))
    with open(path, 'w') as f:
        f.writelines(lines)

np.random.seed(1)

x, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)

samples, factors = x.shape
output_clases = 1
nn = NeuralNetwork(layers=[factors, factors / 10, 1], learning_rate=0.2, epochs=80000)
nn.fit(x, y)
y_test = nn.predict(test)

write_answer('answer.csv', y_test, test_id)

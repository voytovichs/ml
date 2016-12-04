import itertools
import os
import random

import numpy as np


class FeedforwardNetwork:
    def __init__(self, layers, learning_rate, epochs, mini_batch):
        self.fitted_ = False
        self.biases_ = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights_ = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

        self.num_layers_ = len(layers)
        self.layers_ = layers
        self.learning_rate_ = learning_rate
        self.epochs_ = epochs
        self.mini_batch_size_ = mini_batch

        print('Initialized Network:')
        print(' Layers={}'.format(self.layers_))
        print(' LearningRate={}'.format(self.learning_rate_))
        print(' Epochs={}'.format(self.epochs_))
        print(' MiniBatchSize={}'.format(self.mini_batch_size_))

    def update_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases_]
        nabla_w = [np.zeros(w.shape) for w in self.weights_]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights_ = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights_, nabla_w)]
        self.biases_ = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases_, nabla_b)]

    def gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, learning_rate)
                print("Epoch {0} complete".format(j))

    def feedforward(self, a):
        for b, w in zip(self.biases_, self.weights_):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def fit(self, x, y):
        training_data = zip(x, y)
        self.gradient_descent(training_data, self.epochs_, self.mini_batch_size_, self.learning_rate_)
        self.fitted_ = True

    def predict(self, test_data):
        if self.fitted_:
            raise Exception('Call fit first!')
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def exclude_col(matrix, *args):
    return np.delete(matrix, args, axis=1)


def exclude_row(matrix, *args):
    return np.delete(matrix, args, axis=0)


def read_x(path, exclude_y=True, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    _row, col = data.shape
    sub = 1 if exclude_y else 0
    return np.matrix(exclude_col(data, 0, col - sub)), np.array(data.T[0])  # data, id's


def normalize(x, mean, std):
    new_x = []
    for col, m, s in zip(x.T.A, mean, std):
        new_x.append((col - m) / s)
    return np.matrix(new_x).T


def get_mean_and_std(x):
    mean = []
    std = []
    for row in x.T.A:
        mean.append(row.mean())
        std.append(row.std())
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
    tmp = 'haha.csv'
    np.savetxt(tmp, data, fmt='%f', header='id,label', delimiter=',', comments='')
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1:f}\n'.format(ids[i - 1], float(lines[i]))
    with open(path, 'w') as f:
        f.writelines(lines)


x, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)
x, test = preprocess(x, test)

factors = x.shape[1]
output_clases = 2
nn = FeedforwardNetwork(layers=[factors, 924 * 42, 5 * 42, 42, output_clases], learning_rate=42, epochs=42 * 42,
                        mini_batch=42 * 42 * 42)
nn.fit(x, y)
y_test = nn.predict(test)

write_answer('answer.csv', y_test, test_id)

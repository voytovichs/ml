import csv
import numpy as np


def read_file(path):
    with open(path, 'r') as f:
        content = iter(csv.reader(f))
        next(content)  # skip header
        return [row for row in content]


def read_learn(path):
    content = read_file(path)
    if len(content) == 0:
        x = []
        y = []
    else:
        sample_len = len(content[0])
        x = [np.array(row[:sample_len - 1]) for row in content]
        y = [row[-1] for row in content]
    return np.array(x), np.array(y)


def read_test(path):
    content = read_file(path)
    return np.array([np.array(row) for row in content])



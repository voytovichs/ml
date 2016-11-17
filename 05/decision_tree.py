from collections import namedtuple

import itertools
import numpy as np
import os


class DecisionTree:
    def __init__(self, min_leaf_members=100, score_threshold=0.1, split_bounds=20):
        self.fit_called_ = False
        self.min_leaf_members_ = min_leaf_members
        self.score_threshold = score_threshold
        self.tree_ = {}  # Node number -> split condition
        self.SplitCondition_ = namedtuple('SplitCondition', 'feature threshold')
        self.Leaf_ = namedtuple('Leaf', 'zeros ones')
        self.split_bounds_ = split_bounds

    def partition_stop_condition__(self, labels, partition=(float(6) / float(7))):
        one_members = sum(labels)
        return one_members >= partition * len(labels) or one_members <= (1 - partition) * len(labels)

    def build_stop_condition_(self, labels):
        if len(labels) <= self.min_leaf_members_:
            return True
        return self.partition_stop_condition__(labels)

    def select_split_condition_uniform_(self, data, labels):
        best_sc = None
        best_score = float('-inf')

        if len(data) == 0:
            raise Exception('To create split condition \'data\' must not be emtpy')
        m = data.shape[1]

        for feature in range(m):
            mapped = map(lambda x: x[feature], data.A)
            min_sample = np.min(mapped)
            max_sample = np.max(mapped)

            step = (max_sample - min_sample) / (self.split_bounds_ + 1)
            bounds = [(min_sample + step)]
            for i in range(1, self.split_bounds_):
                bounds.append(bounds[i - 1] + step)

            for bound in bounds:
                sc = self.SplitCondition_(feature, bound)
                a, a_lbs, b, b_lbs = self.split_on_condition_(sc, data, labels)
                score = self.metric_value_(a_lbs, b_lbs)
                if (score > best_score):
                    best_score = score
                    best_sc = sc
        return best_sc

    def split_on_condition_(self, split_condition, data, labels):
        a, b = [], []
        a_lbs, b_lbs = [], []

        for (entry, label) in zip(data.A, labels):
            if entry[split_condition.feature] <= split_condition.threshold:
                a.append(entry)
                a_lbs.append(label)
            else:
                b.append(entry)
                b_lbs.append(label)

        return np.matrix(a), a_lbs, np.matrix(b), b_lbs

    def create_leaf_(self, labels):
        ones = sum(labels)
        zeros = len(labels) - ones
        return self.Leaf_(zeros, ones)

    def build_tree_(self, node_num, last_score, data, labels):

        if self.build_stop_condition_(labels):
            self.tree_[node_num] = self.create_leaf_(labels)
            return

        sc = self.select_split_condition_uniform_(data, labels)
        self.tree_[node_num] = sc
        a, a_lbs, b, b_lbs = self.split_on_condition_(sc, data, labels)

        score = self.metric_value_(a_lbs, b_lbs)
        if abs(last_score - score) < self.score_threshold:
            self.tree_[node_num] = self.create_leaf_(labels)
            return

        self.build_tree_(node_num * 2, score, a, a_lbs)
        self.build_tree_(node_num * 2 + 1, score, b, b_lbs)

    def entropy__(self, labels):
        if len(labels) == 0:
            return 0

        p = sum(labels) / float(len(labels))
        q = 1 - p

        if p == 0 or q == 0:
            return 0
        return - (p * np.log(p)) - (q * np.log(q))

    def metric_value_(self, a_lbs, b_lbs):
        return -self.entropy__(a_lbs) - self.entropy__(b_lbs)

    def fit(self, data, labels):
        self.build_tree_(1, float('-inf'), data, labels)
        self.fit_called_ = True

    def predict_(self, node_num, x):
        node = self.tree_[node_num]

        if isinstance(node, self.Leaf_):
            return node.zeros / float(node.zeros + node.ones)

        if not isinstance(node, self.SplitCondition_):
            raise Exception('The node {0} is neither a leaf nor a split condition: {1}'
                            .format(node_num, node))

        if x[node.feature] <= node.threshold:
            return self.predict_(node_num * 2, x)
        else:
            return self.predict_(node_num * 2 + 1, x)

    def predict(self, data):
        if not self.fit_called_:
            raise Exception('Call fit first')
        labels = []
        for sample in data.A:
            label = self.predict_(1, sample)
            labels.append(label)
        return np.array(labels)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def read_x(path, exclude_y=True, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)

    sub = 1 if exclude_y else 0  # Get rid of labels

    _row, col = data.shape
    print('shape {} {}'.format(_row, col))

    return np.matrix(exclude_col(data, 0, col - sub)), np.array(data.T[0])  # data, id's


def read_y(path, n=None):
    with open(path, 'r') as f:
        if n is None:
            data = np.genfromtxt(f, delimiter=',', skip_header=True)
        else:
            data = np.genfromtxt(itertools.islice(f, 0, n), delimiter=',', skip_header=True)
    row, col = data.shape
    return np.array([data[i, col - 1] for i in range(row)]), np.array(data.T[0])


def write_answer(path, data, ids):
    tmp = 'haha.csv'
    np.savetxt(tmp, data, fmt='%d', header='id,label', delimiter=',', comments='')
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1:f}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as f:
        f.writelines(lines)


x, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)

tree = DecisionTree(min_leaf_members=100, split_bounds=20)
tree.fit(x, y)
y_test = tree.predict(test)
write_answer('answer.csv', y_test, test_id)
print('Done')

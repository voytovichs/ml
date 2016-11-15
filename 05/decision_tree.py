import numpy as np
from collections import namedtuple


class DecisionTree:
    def __init__(self):
        self.fit_called_ = False
        self.tree_ = {}  # Node number -> split condition
        self.SplitCondition_ = collections.namedtuple('SplitCondition', 'feature threshold')

    def partition_stop_condition_(self, labels, partition=(float(5) / float(6))):
        zero_members = len(filter(labels, lambda a: a == 0))
        return zero_members >= partition * len(labels) or zero_members <= (1 - partition) * len(labels)

    def build_stop_condition_(self, labels):
        return self.partition_stop_condition_(labels)

    def select_split_condition_(self, data, labels):
        raise Error('Not implemented')

    def split_on_condition_(self, split_condition, data, labels):
        a, b = [], []
        a_lbs, b_lbs = [], []
        for (entry, label) in zip(data, labels):
            if entry[split_condition.feature] <= split_condition.threshold:
                a.append(entry)
                a_lbs.append(label)
            else:
                b.append(entry)
                b_lbs.append(label)
        return a, a_lbs, b, b_lbs

    def build_tree_(self, node_num, data, labels):
        if self.build_stop_condition_(labels):
            return
        sc = self.select_split_condition_(data, labels)
        self.tree_[node_num] = sc
        a, a_lbs, b, b_lbs = self.split_on_condition_(sc, data, labels)
        self.build_tree_(node_num * 2, a, a_lbs)
        self.build_tree_(node_num * 2 + 1, b, b_lbs)

    def metric_value_(self, labels):
        raise Error('Not implemented')

    def fit(self, data, labels):
        self.build_tree_(1, data, labels)
        self.fit_called_ = True

    def predict(self, data):
        if self.fit_called_:
            raise Error('Call fit first')
        return [0] ** len(data)


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
        lines[i] = '{0:g},{1}'.format(ids[i - 1], lines[i])
    with open(path, 'w') as f:
        f.writelines(lines)


x, x_id = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id = read_x('test.csv', exclude_y=False)

tree = DecisionTree()
tree.fit(x, y)
y_test = tree.predict(test)
write('answer.csv', y_test, test_id)

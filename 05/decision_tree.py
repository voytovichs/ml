from collections import namedtuple
import itertools
import numpy as np
import os


class DecisionTree:
    def __init__(self,
                 split_bound_number=20,
                 stop_leaf_size=100,
                 stop_score_threshold=float('-inf'),
                 stop_partition=(float(49) / float(50))):

        self.stop_leaf_size_ = stop_leaf_size
        self.stop_score_threshold_ = stop_score_threshold
        self.split_bounds_ = split_bound_number
        self.stop_partition_ = stop_partition

        self.fit_called_ = False
        self.tree_ = {}  # Node number -> split condition
        self.SplitCondition_ = namedtuple('SplitCondition', 'feature threshold')
        self.Leaf_ = namedtuple('Leaf', 'zeros ones')
        self.feature_bounds__ = None

        print('Initialized tree:')
        print(' StopLeafSize={}'.format(self.stop_leaf_size_))
        print(' StopScoreThreshold={}'.format(self.stop_score_threshold_))
        print(' SplitBoundsNumber={}'.format(self.split_bounds_))
        print(' StopPartition={}'.format(self.stop_partition_))

    ''' Variative part '''

    def entropy__(self, labels):
        if len(labels) == 0:
            return 0

        p = sum(labels) / float(len(labels))
        q = 1 - p

        if p == 0 or q == 0:
            return 0
        return - (p * np.log2(p)) - (q * np.log2(q))

    def get_feature_split_bounds__(self, data):
        result = []
        m = data.shape[1]
        for feature in range(m):
            mapped = map(lambda x: x[feature], data.A)
            min_sample = np.min(mapped)
            max_sample = np.max(mapped)

            step = (max_sample - min_sample) / (self.split_bounds_ + 1)
            bounds = [(min_sample + step)]

            for i in range(1, self.split_bounds_):
                bounds.append(bounds[i - 1] + step)

            bounds.append(np.median(mapped))
            bounds.append(np.mean(mapped))
            result.append(bounds)
        return result

    def select_split_condition_uni__(self, data, labels, bounds):
        best_sc = None
        best_score = float('-inf')

        if len(data) == 0:
            raise Exception('To create split condition \'data\' must not be emtpy')
        m = data.shape[1]

        for feature in range(m):
            for bound in bounds[feature]:
                sc = self.SplitCondition_(feature, bound)
                a, a_lbs, b, b_lbs = self.split_on_condition_(sc, data, labels)
                score = self.metric_score_(a_lbs, b_lbs)
                if (score > best_score):
                    best_score = score
                    best_sc = sc
        return best_sc

    def partition_stop_condition__(self, labels, partition):
        one_members = sum(labels) / len(labels)
        return one_members >= partition or one_members <= (1 - partition)

    ''' Non-variative part '''

    def metric_score_(self, a_lbs, b_lbs):
        entire_set_e = self.entropy__(a_lbs + b_lbs)
        left_w = len(a_lbs) / float(len(a_lbs) + len(b_lbs))
        right_w = len(b_lbs) / float(len(a_lbs) + len(b_lbs))
        left_leaf_e = self.entropy__(a_lbs) * left_w
        right_leaf_e = self.entropy__(b_lbs) * right_w
        return entire_set_e - (left_leaf_e + right_leaf_e)

    def build_stop_condition_(self, labels):
        if self.stop_leaf_size_ is not None and len(labels) <= self.stop_leaf_size_:
            return True
        return self.partition_stop_condition__(labels, partition=self.stop_partition_)

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

        sc = self.select_split_condition_uni__(data, labels, self.feature_bounds__)
        self.tree_[node_num] = sc
        a, a_lbs, b, b_lbs = self.split_on_condition_(sc, data, labels)

        if len(a_lbs) < self.stop_leaf_size_ or len(b_lbs) < self.stop_leaf_size_:
            self.tree_[node_num] = self.create_leaf_(labels)
            return

        score = self.metric_score_(a_lbs, b_lbs)
        if abs(last_score - score) < self.stop_score_threshold_:
            self.tree_[node_num] = self.create_leaf_(labels)
            return

        print('building node #{} ...'.format(node_num * 2))
        self.build_tree_(node_num * 2, score, a, a_lbs)

        print('building node #{}...'.format(node_num * 2 + 1))
        self.build_tree_(node_num * 2 + 1, score, b, b_lbs)

    def fit(self, data, labels):
        self.feature_bounds__ = self.get_feature_split_bounds__(data)
        self.build_tree_(1, float('-inf'), data, labels)
        self.fit_called_ = True

    def predict_traversing_(self, node_num, x):
        node = self.tree_[node_num]

        if isinstance(node, self.Leaf_):
            return node.zeros / float(node.zeros + node.ones)

        if not isinstance(node, self.SplitCondition_):
            raise Exception('The node {0} is neither a leaf nor a split condition: {1}'
                            .format(node_num, node))

        if x[node.feature] <= node.threshold:
            return self.predict_traversing_(node_num * 2, x)
        else:
            return self.predict_traversing_(node_num * 2 + 1, x)

    def predict(self, data):
        if not self.fit_called_:
            raise Exception('Call fit first')
        labels = []
        for sample in data.A:
            label = self.predict_traversing_(1, sample)
            labels.append(label)
        return np.array(labels)


def exclude_col(X, *args):
    return np.delete(X, args, axis=1)


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def map_char_to_num(char, mapping):
    if char in mapping:
        return mapping[char]
    index = len(mapping)
    mapping[char] = index
    return index


def read_x(path, exclude_y=True, n=None, mapping=None):
    data = []
    if mapping is None:
        mapping = {}

    with open(path, 'r') as f:
        data = f.readlines()[1:]
        data = map(lambda s: s.split(','), data)
        data = list(
            map(lambda row: map(lambda a: float(a) if is_number(a) else map_char_to_num(a, mapping), row), data))

    data = np.matrix(data)
    sub = 1 if exclude_y else 0  # Get rid of labels

    _row, col = data.shape
    return np.matrix(exclude_col(data, 0, col - sub)), np.array(data.T.A[0]), mapping  # data, id's, mapping


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
    np.savetxt(tmp, data, fmt='%f', header='id,label', delimiter=',', comments='')
    with open(tmp, 'r') as f:
        lines = f.readlines()
    os.remove(tmp)
    for i in range(1, len(lines)):
        lines[i] = '{0:g},{1:f}\n'.format(ids[i - 1], float(lines[i]))
    with open(path, 'w') as f:
        f.writelines(lines)


x, x_id, mapping = read_x('learn.csv')
y, y_id = read_y('learn.csv')
test, test_id, _ = read_x('test.csv', exclude_y=False, mapping=mapping)

tree = DecisionTree(stop_leaf_size=30,
                    split_bound_number=30,
                    stop_partition=float(9) / float(10))
tree.fit(np.matrix(x), y)
y_test = tree.predict(np.matrix(test))
write_answer('answer.csv', y_test, test_id)

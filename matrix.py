class Matrix(object):
    def __init__(self, row_len, col_len):
        self.row_len = row_len
        self.col_len = col_len
        self.rows = [[0] * self.col_len] * self.row_len

    def _check_index(self, row, column):
        if row < 0 or row >= self.row_len or column < 0 or column > self.row_len:
            raise IndexError('Index is out of range')

    def __getitem__(self, index):
        row, column = index
        self._check_index(row, column)
        return self.rows[row][column]

    def __setitem__(self, index, value):
        row, column = index
        self._check_index(row, column)
        self.rows[index] = value

    def __delitem__(self, index):
        pass

    def __add__(self, other):
        print('Add called')

    def __mul__(self, other):
        print('Mul called')

    def get_transposed(self):
        pass

    def get_inverted(self):
        if self.rows != self.columns:
            raise ValueError('Non-square matrix cannot be inverted')

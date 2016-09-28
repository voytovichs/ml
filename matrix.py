class Matrix(object):
    def __init__(self, row_len, col_len):
        self.row_len = row_len
        self.col_len = col_len
        self.rows = [[0] * self.col_len] * self.row_len

    def __getitem__(self, index):
        return self.rows[index]

    def __setitem__(self, index, value):
        self.rows[index] = value

    def __delitem__(self, index):
        pass

    def __add__(self, other):
        if self.row_len != other.row_len or self.col_len != other.row_len:
            raise ValueError('Cannot add matrices with different shapes')
        rows = [list(map(sum, zip(self.rows[i], other[i]))) for i in range(self.row_len)]
        result = Matrix(self.row_len, self.col_len)
        result.rows = rows
        return result

    def __mul__(self, other):
        print('Mul called')

    def get_transposed(self):
        pass

    def get_inverted(self):
        if self.rows != self.columns:
            raise ValueError('Non-square matrix cannot be inverted')

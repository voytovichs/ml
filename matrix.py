class Matrix(object):
    def __init__(self, row_len, col_len):
        self.row_n = row_len
        self.col_n = col_len
        self.rows = []
        for i in range(self.row_n):
            self.rows.append([0] * self.col_n)

    def __getitem__(self, index):
        return self.rows[index]

    def __setitem__(self, index, value):
        self.rows[index] = value

    def __delitem__(self, index):
        pass

    def __repr__(self):
        return str(self.rows)

    def __add__(self, other):
        if self.row_n != other.row_n or self.col_n != other.row_n:
            raise ValueError('Matrices do not fit addition shape condition')
        rows = [list(map(sum, zip(self.rows[i], other[i]))) for i in range(self.row_n)]
        result = Matrix(self.row_n, self.col_n)
        result.rows = rows
        return result

    def __mul__(self, other):
        if self.row_n != other.col_n or self.col_n != other.row_n:
            raise ValueError('Matrices do not fit multiplying shape condition')
        m = Matrix(self.row_n, other.col_n)
        for i in range(m.row_n):
            for j in range(m.col_n):
                m[i][j] = sum([self[i][k] * other[k][j] for k in range(len(self[i]))])

        return m

    def get_transposed(self):
        m = Matrix(self.col_n, self.row_n)
        for i in range(self.row_n):
            for j in range(self.col_n):
                m[j][i] = self[i][j]
        return m

    def get_inverted(self):
        if self.rows != self.columns:
            raise ValueError('Non-square matrix cannot be inverted')
        

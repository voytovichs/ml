class Matrix(object):
    def __init__(self, row_len, col_len, initial=None):
        self.row_n = row_len
        self.col_n = col_len
        self.rows = []
        if not initial:
            for i in range(self.row_n):
                self.rows.append([0] * self.col_n)
        else:
            for i in range(self.row_n):
                self.rows.append(initial[i][:])

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

    def _get_identity_matrix(self):
        if self.row_n != self.col_n:
            raise ValueError("Cannot compute identity matrix for non-square shape")
        m = Matrix(self.row_n, self.row_n)
        for i in range(self.row_n):
            m[i][i] = 1
        return m

    def _pivotize(self):
        ID = self._get_identity_matrix()
        n = self.row_n
        for j in range(n):
            row = max(range(j, n), key=lambda i: abs(self[i][j]))
            if j != row:
                ID[j], ID[row] = ID[row], ID[j]
        return ID

    def _get_LU(self):
        if self.row_n != self.col_n:
            raise ValueError('Cannot find LU decomposition for non-square matrix')
        n = self.row_n
        L = self._get_identity_matrix()
        U = Matrix(n, n)
        P = self._pivotize()
        A2 = P * self
        for j in range(n):
            L[j][j] = 1
            for i in range(j + 1):
                s1 = sum(U[k][j] * L[i][k] for k in range(i))
                U[i][j] = A2[i][j] - s1
            for i in range(j, n):
                s2 = sum(U[k][j] * L[i][k] for k in range(i))
                L[i][j] = (A2[i][j] - s2) / U[j][j]
        return L, U

    def get_inverted(self):
        if self.row_n != self.col_n:
            raise ValueError('Non-square matrix cannot be inverted')

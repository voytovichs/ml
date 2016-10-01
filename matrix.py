class Matrix(object):
    def __init__(self, init_data):
        if isinstance(init_data, tuple):
            n, m = init_data
            self.row_n = n
            self.col_n = m
            self.rows = []
            for i in range(self.row_n):
                self.rows.append([0] * self.col_n)
        else:
            array = init_data
            n = len(array)
            m = 0
            if n != 0:
                if isinstance(array[0], list):
                    m = len(array[0])
                else:
                    m = 1
            self.row_n = n
            self.col_n = m
            self.rows = []
            if isinstance(array[0], list):
                for i in range(n):
                    self.rows.append(array[i][:])
            else:
                for i in range(n):
                    self.rows.append(array[i])


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
            raise ValueError('Matrices do not fit shape condition')
        rows = [list(map(sum, zip(self.rows[i], other[i]))) for i in range(self.row_n)]
        result = Matrix((self.row_n, self.col_n))
        result.rows = rows
        return result

    def __mul__(self, other):
        if self.row_n != other.col_n or self.col_n != other.row_n:
            raise ValueError('Matrices do not fit shape condition')
        m = Matrix((self.row_n, other.col_n))
        for i in range(m.row_n):
            for j in range(m.col_n):
                m[i][j] = sum([self[i][k] * other[k][j] for k in range(len(self[i]))])
        return m

    def get_transposed(self):
        m = Matrix((self.col_n, self.row_n))
        for i in range(self.row_n):
            for j in range(self.col_n):
                m[j][i] = self[i][j]
        return m

    def _get_identity_matrix(self):
        if self.row_n != self.col_n:
            raise ValueError("Cannot compute identity matrix for non-square shape")
        m = Matrix((self.row_n, self.row_n))
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
        U = Matrix((n, n))
        P = self._pivotize()
        A2 = P * self
        for j in range(n):
            L[j][j] = 1
            for i in range(j + 1):
                s1 = sum(U[k][j] * L[i][k] for k in range(i))
                U[i][j] = A2[i][j] - s1
            for i in range(j, n):
                s2 = sum(U[k][j] * L[i][k] for k in range(j))
                L[i][j] = (A2[i][j] - s2) / U[j][j]
        return L, U

    def get_inverted(self):
        if self.row_n != self.col_n:
            raise ValueError('Non-square matrix cannot be inverted')
        L, U = self._get_LU()
        n = self.row_n
        R = self
        # Ly=I, forward substitution
        for k in range(n):
            for i in range(k + 1, n):
                for j in range(n):
                    R[i][j] -= R[k][j] * L[i][k]
        # U*X=y, backward substitution
        for k in range(n - 1, -1, -1):
            for j in range(n):
                R[k][j] /= U[k][k]
            for i in range(k):
                for j in range(n):
                    R[i][j] -= R[k][j] * U[i][k]
        return R


def mult_matrix_test():
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    m2 = Matrix([[7, 8], [9, 10], [11, 12]])
    m3 = m1 * m2
    m3_expected = [[58, 64], [139, 154]]
    for i in range(len(m3_expected)):
        for j in range(len(m3_expected[0])):
            assert m3[i][j] == m3_expected[i][j]


def lu_decomposition_test():
    m = Matrix([[1, 0, 2], [2, -1, 3], [4, 1, 8]])
    L, U = m._get_LU()
    LU = L * U
    print(L)
    print(U)
    for i in range(m.row_n):
        for j in range(m.col_n):
            # lines order may change, compare LU with m manually
            assert m[i][j] == LU[i][j], 'm[{0}][{1}](={2}) != LU[{0}][{1}](={3})'.format(i, j, m[i][j], LU[i][j])


def inverted_matrix_test():
    m = Matrix([[1, 0, 2], [2, -1, 3], [4, 1, 8]])
    m_inv = m.get_inverted()
    I = m * m_inv
    print(I)
    for i in range(m.row_n):
        for j in range(m.col_n):
            if i == j:
                assert abs(I[i][j] - 1) < 0.001, 'I[{0}][{1}] != 1 (=={2})'.format(i, j, I[i][j])
            else:
                assert abs(I[i][j]) < 0.001, 'I[{0}][{1}] != 0 (=={2})'.format(i, j, I[i][j])

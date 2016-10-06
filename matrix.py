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
            if not isinstance(array[0], list):
                self.row_n = len(array)
                self.col_n = 1
                self.rows = []
                for i in range(self.row_n):
                    self.rows.append([array[i]])
            else:
                n = len(array)
                m = len(array[0])
                self.row_n = n
                self.col_n = m
                self.rows = []
                for i in range(n):
                    self.rows.append(array[i][:])

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
        if self.col_n != other.row_n:
            raise ValueError('Matrices do not fit shape condition. {0} X {1}'.format(self.shape(), other.shape()))
        m = Matrix((self.row_n, other.col_n))
        for i in range(m.row_n):
            for j in range(m.col_n):
                m[i][j] = sum([self[i][k] * other[k][j] for k in range(len(self[i]))])
        return m

    def swap(self, i, j):
        tmp = self.rows[i]
        self.rows[i] = self.rows[j]
        self.rows[j] = tmp
        return self

    def shape(self):
        return self.row_n, self.col_n

    def get_lup_decomposition(self):
        if self.row_n != self.col_n:
            raise ValueError('Cannot find LUP decomposition for non-square matrix')
        n = self.row_n
        pi = [i for i in range(n)]
        lu = Matrix(self.rows)
        for k in range(n):
            p = 0
            for i in range(k, n):
                if abs(lu[i][k]) > p:
                    p = lu[i][k]
                    k_p = i
            if p == 0:
                raise ValueError('Cannot find LUP decomposition of singular matrix')
            tmp = pi[k]
            pi[k] = pi[k_p]
            pi[k_p] = tmp
            lu.swap(k, k_p)
            for i in range(k + 1, n):
                lu[i][k] /= lu[k][k]
                for j in range(k + 1, n):
                    lu[i][j] -= lu[i][k] * lu[k][j]
        L = Matrix(lu.rows)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    L[i][j] = 1
                else:
                    L[i][j] = 0
        U = lu
        for i in range(n):
            for j in range(i):
                U[i][j] = 0
        return L, U, pi

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

    def lup_solve(self, L, U, pi, b):
        n = self.row_n
        y = [0] * n
        x = [0] * n
        for i in range(n):
            y[i] = b[pi[i]] - sum([L[i][j] * y[j] for j in range(i)])
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum([U[i][j] * x[j] for j in range(i + 1, n)])) / U[i][i]
        return x

    def get_column(self, i):
        if i < 0 or i >= self.col_n:
            raise ValueError('Column with number {} does not exists'.format(i))
        return [self[j][i] for j in range(self.row_n)]

    def get_with_exluded_column(self, i):
        new_rows = self.get_transposed().rows
        del(new_rows[i])
        return Matrix(new_rows).get_transposed()

    def get_inverted(self):
        if self.row_n != self.col_n:
            raise ValueError('Non-square matrix cannot be inverted')
        L, U, pi = self.get_lup_decomposition()
        n = self.row_n
        inverted = []
        for i in range(n):
            b = [0] * n
            b[i] = 1
            inverted.append(self.lup_solve(L, U, pi, b))
        return Matrix(inverted).get_transposed()


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
    L, U, pi = m.get_lup_decomposition()
    P = Matrix((3, 3))
    for i in range(len(pi)):
        P[i][pi[i]] = 1
    LU = L * U
    Pm = P * m
    for i in range(m.row_n):
        for j in range(m.col_n):
            assert Pm[i][j] == LU[i][j], 'm[{0}][{1}](={2}) != LU[{0}][{1}](={3})'.format(i, j, m[i][j], LU[i][j])


def lup_solve_test():
    m = Matrix([[1, 2, 0], [3, 5, 4], [5, 6, 3]])
    L, U, pi = m.get_lup_decomposition()
    x = m.lup_solve(L, U, pi, [0.1, 12.5, 10.3])
    x_expected = [0.5, -0.2, 3]
    for i in range(len(x)):
        assert abs(x[i] - x_expected[i]) < 0.001


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

# mult_matrix_test()
# lu_decomposition_test()
# lup_solve_test()
# inverted_matrix_test()

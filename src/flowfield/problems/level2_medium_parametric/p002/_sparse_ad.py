"""Small vector forward AD algebra for the sparse P002 discrete residual.

Rows correspond to array elements, columns to independent variables. Branches
use the primal value (piecewise derivatives); no CFD solves are differentiated.
"""

import numpy as np
from scipy import sparse


class Dual:
    __array_priority__ = 1000

    def __init__(self, value, jacobian):
        self.v = np.asarray(value, dtype=float).reshape(-1)
        self.j = sparse.csr_matrix(jacobian)
        if self.j.shape[0] != self.v.size:
            raise ValueError('AD value/Jacobian row mismatch')

    def constant(self, value):
        value = np.asarray(value, dtype=float).reshape(-1)
        return Dual(value, sparse.csr_matrix((value.size, self.j.shape[1])))

    def __getitem__(self, key):
        ids = np.arange(self.v.size)[key]
        ids = np.asarray(ids).reshape(-1)
        return Dual(self.v[ids], self.j[ids])

    def _pair(self, other):
        other = other if isinstance(other, Dual) else self.constant(other)
        a, b = self, other
        n = max(a.v.size, b.v.size)
        if a.v.size == 1 and n != 1:
            a = a[np.zeros(n, dtype=int)]
        if b.v.size == 1 and n != 1:
            b = b[np.zeros(n, dtype=int)]
        if a.v.size != b.v.size:
            raise ValueError('Only scalar/vector AD broadcasting is supported')
        return a, b

    def __add__(self, other):
        a, b = self._pair(other)
        return Dual(a.v+b.v, a.j+b.j)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.j)

    def __sub__(self, other):
        return self+-other

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        a, b = self._pair(other)
        return Dual(a.v*b.v, a.j.multiply(b.v[:, None])+b.j.multiply(a.v[:, None]))

    __rmul__ = __mul__

    def __truediv__(self, other):
        a, b = self._pair(other)
        return Dual(a.v/b.v, a.j.multiply((1/b.v)[:, None])-
                    b.j.multiply((a.v/b.v**2)[:, None]))

    def __rtruediv__(self, other):
        return self.constant(other)/self

    def __pow__(self, power):
        return Dual(self.v**power, self.j.multiply((power*self.v**(power-1))[:, None]))

    def exp(self):
        value = np.exp(self.v)
        return Dual(value, self.j.multiply(value[:, None]))

    def log(self):
        return Dual(np.log(self.v), self.j.multiply((1/self.v)[:, None]))

    def clip(self, lower, upper=np.inf):
        return Dual(np.clip(self.v, lower, upper),
                    self.j.multiply(((self.v > lower) & (self.v < upper))[:, None]))

    def absolute(self):
        return Dual(np.abs(self.v), self.j.multiply(np.sign(self.v)[:, None]))

    def sum(self):
        return linear(sparse.csr_matrix(np.ones((1, self.v.size))), self)


def linear(matrix, value):
    return Dual(matrix@value.v, matrix@value.j)


def concatenate(values):
    return Dual(np.concatenate([v.v for v in values]), sparse.vstack([v.j for v in values], format='csr'))


def select(mask, a, b):
    if not isinstance(a, Dual):
        a = b.constant(a)
    a, b = a._pair(b)
    mask = np.asarray(mask, dtype=bool)
    return Dual(np.where(mask, a.v, b.v),
                a.j.multiply(mask[:, None])+b.j.multiply((~mask)[:, None]))


def scatter(value, indices, size):
    indices = np.asarray(indices, dtype=int)
    matrix = sparse.csr_matrix((np.ones(len(indices)), (indices, np.arange(len(indices)))),
                               shape=(size, len(indices)))
    return linear(matrix, value)

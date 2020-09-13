import numpy as np
cimport numpy as np

cdef class Math:
    # @staticmethod
    # cdef np.ndarray[np.float_t, ndim=2] polyval2d(np.ndarray[np.float_t, ndim=2] p, np.ndarray[np.float_t, ndim=1] x):
    #     """
    #     Functionality similar to numpy.polyval, except now p can be multiple poly1d instances - one in each row,
    #     while enjoying matrix-operations efficiency
    #     :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
    #     :param x: a 1d numpy array [N] of samples
    #     :return: a 2d numpy array [MxN] of polynom values for each poly1d instance and sample
    #     """
    #     m = len(p)
    #     l = p.shape[1]
    #     n = len(x)
    #
    #     y = np.zeros(shape=[m, n])
    #     for i in range(l):
    #         y = np.einsum('ij,j->ij', y, x) + np.repeat(p[:, i, np.newaxis], n, axis=1)
    #
    #     return y

    @staticmethod
    def zip_polyval2d(p, x):
        return Math.czip_polyval2d(p, x)

    @staticmethod
    cdef np.ndarray[np.float_t, ndim=2] czip_polyval2d(np.ndarray[np.float_t, ndim=2] p, np.ndarray[np.float_t, ndim=2] x):
        """
        Functionality similar to numpy.polyval, except now p holds multiple poly1d instances - one in each row,
        and x holds in each row a vector of values to assign to the corresponding poly1d in p.
        this enjoys matrix-operations efficiency
        :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
        :param x: a 2d numpy array [MxN] of samples
        :return: a 2d numpy array [MxN] of polynomial values for each poly1d instance and sample
        """
        assert p.shape[0] == x.shape[0], 'number of values and polynomials is not equal'
        cdef int m = p.shape[0]
        cdef int l = p.shape[1]
        cdef int n = x.shape[1]

        cdef np.ndarray[np.float_t, ndim=2] y = np.zeros(shape=[m, n])
        cdef int i
        for i in range(l):
            y = np.einsum('ij,ij->ij', y, x) + np.repeat(p[:, i, np.newaxis], n, axis=1)

        return y

    @staticmethod
    def polyder2d(p, m):
        return Math.cpolyder2d(p, m)

    @staticmethod
    cdef np.ndarray[np.float_t, ndim=2] cpolyder2d(np.ndarray[np.float_t, ndim=2] p, int m):
        """
        Functionality similar to numpy.polyval, except now p can be multiple poly1d instances - one in each row,
        while enjoying matrix-operations efficiency
        :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
        :param m: derivative degree
        :return: a 2d numpy array [MxN] of polynom values for each poly1d instance and sample
        """
        cdef:
            int n
            np.ndarray[np.float_t, ndim=2] y
            np.ndarray[np.float_t, ndim=2] val

        n = p.shape[1] - 1
        y = p[:, :-1] * np.arange(n, 0, -1)
        if m == 0:
            val = p
        else:
            val = Math.cpolyder2d(y, m - 1)
        return val
    #
    # @staticmethod
    # def roots(p):
    #     """
    #     Return the roots of polynomials with coefficients given in the rows of p.
    #     The values in each row of the matrix `p` are coefficients of a polynomial.
    #     If the length of a row in `p` is n+1 then the polynomial is described by:
    #         p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    #     :param p: A matrix of size (num_of_poly X (poly_degree+1)) which contains polynomial coefficients.
    #               num_of_poly has to be greater than 1.
    #     :return: A matrix containing the roots of the polynomials (a set of roots in each row corresponding
    #             to the polynomial in the input matrix) [ndarray]
    #
    #     """
    #     n = p.shape[-1]
    #     A = np.zeros(p.shape[:1]+(n-1, n-1), float)
    #     A[..., 1:, :-1] = np.eye(n-2)
    #     A[..., 0, :] = -p[..., 1:]/p[..., None, 0]
    #     return np.linalg.eigvals(A)
    #
    # @staticmethod
    # def solve_quadratic(p):
    #     """
    #     Find the roots of a quadratic equation
    #     :param p: a 2d numpy array [Mx3] having in each of the M rows the 3 polynomial coefficients vector [a, b, c]
    #     :return: a 2d numpy array [Mx2] of roots for each poly1d instance, or None if no root exists. Smaller root is at index 0
    #     """
    #
    #     a, b, c = np.hsplit(p, 3)
    #     half_b = b * 0.5
    #     discriminant = half_b * half_b - a * c
    #     valid_roots = np.where(discriminant >= 0)[0]
    #     roots = np.full((p.shape[0], 2), np.nan)
    #     sqrt_disc = np.sqrt(discriminant[valid_roots])
    #     roots[valid_roots] = np.c_[-half_b[valid_roots] - sqrt_disc, -half_b[valid_roots] + sqrt_disc] / a[valid_roots]
    #     return roots

import numpy as np
cimport numpy as np
import cython

cdef class Poly1D(object):
    @staticmethod
    def zip_polyval_with_derivatives(np.ndarray poly_coefs, np.ndarray time_samples):
        return Poly1D._zip_polyval_with_derivatives(poly_coefs, time_samples)

    @staticmethod
    cdef np.ndarray[np.float_t, ndim=3] _zip_polyval_with_derivatives(np.ndarray[np.float_t, ndim=2] poly_coefs, np.ndarray[np.float_t, ndim=2] time_samples):
        """
        For n-th position polynomial and k-th element in n-th row of time-samples matrix (where n runs on all
        polynomials), it generates 3 values:
          1. position (evaluation of the polynomial)
          2. velocity (evaluation of the 1st derivative of the polynomial)
          3. acceleration (evaluation of the 2st derivative of the polynomial)
        :param poly_coefs: 2d numpy array [NxL] of the (position) polynomials coefficients, where
         each row out of the N is a different polynomial and contains L coefficients
        :param time_samples: 2d numpy array [NxK] of the time stamps for the evaluation of the polynomials
        :return: 3d numpy array [N,K,3] with the following dimensions:
            [position value, velocity value, acceleration value]
        """
        cdef:
            np.ndarray[np.float_t, ndim=2] x_vals
            np.ndarray[np.float_t, ndim=2] x_dot_vals
            np.ndarray[np.float_t, ndim=2] x_dotdot_vals

        x_vals = Math.czip_polyval2d(poly_coefs, time_samples)
        x_dot_vals = Math.czip_polyval2d(Math.cpolyder2d(poly_coefs, m=1), time_samples)
        x_dotdot_vals = Math.czip_polyval2d(Math.cpolyder2d(poly_coefs, m=2), time_samples)

        return np.dstack((x_vals, x_dot_vals, x_dotdot_vals))


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
    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def zip_polyval2d(np.ndarray[np.float_t, ndim=2] p, np.ndarray[np.float_t, ndim=2] x):
        return Math.czip_polyval2d(p, x)

    @staticmethod
    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
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
        for i in prange(l):
            y = np.einsum('ij,ij->ij', y, x) + np.repeat(p[:, i, np.newaxis], n, axis=1)

        return y

    @staticmethod
    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def polyder2d(np.ndarray[np.float_t, ndim=2] p, int m):
        return Math.cpolyder2d(p, m)

    @staticmethod
    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
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

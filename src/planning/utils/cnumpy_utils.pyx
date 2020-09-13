import numpy as np
cimport numpy as np

from decision_making.src.planning.types import Limits, LIMIT_MIN, LIMIT_MAX


@staticmethod
cdef np.ndarray[np.float_t] cartesian_product_matrix_rows(np.ndarray[np.float_t] mat1, np.ndarray[np.float_t] mat2):
    """
    this function takes two 2D numpy arrays (treats them as matrices) and builds a new matrix whose rows are the
    cartesian products of the rows of the two original matrices
    :param mat1: 2D numpy array [r1 x c1]
    :param mat2: 2D numpy array [r2 x c2]
    :return: 2D numpy array [r1*r2 x (c1+c2)]
    """
    cdef tuple mat1_shape_for_tile = tuple([1] * len(mat1.dimensions))
    mat1_shape_for_tile[0] = len(mat1)
    return np.concatenate((np.repeat(mat1, len(mat2), axis=0), np.tile(mat2, mat1_shape_for_tile)),
                          axis=len(mat1.dimensions) - 1)

@staticmethod
cdef np.ndarray[np.float_t] row_wise_normal(np.ndarray[np.float_t] mat):
    """
    Utility function that takes a 2D matrix of shape [N, 2] and computes a normal vector for each row, assuming
     mat[i, 0] and mat[i, 1] are (x_i, y_i) coordinates of a vector.
    :param mat: 2D numpy array with shape [N, 2]
    :return: 2D numpy array with shape [N, 2] of normal vectors
    """
    return np.c_[-mat[:, 1], mat[:, 0]]

@staticmethod
cdef np.ndarray[np.uint8_t] is_in_limits(np.ndarray[np.float_t] arr, list limits):
    """
    tests if values of arr are in the limit [lb, ub]
    :param arr: any tensor shape
    :param limits: Limits object - 1D numpy array of [lower_bound, upper_bound]
    :return: tensor of boolean values of the shape of <arr>
    """
    return np.logical_and(arr >= limits[LIMIT_MIN], arr <= limits[LIMIT_MAX])

@staticmethod
cdef np.ndarray[np.uint8_t] zip_is_in_limits(np.ndarray[np.float_t] arr, np.ndarray[np.float_t] limits):
    """
    tests if values of arr are in the limit [lb, ub]
    :param arr: any tensor shape whose first axis correspond to the first axis in limits
    :param limits: Limits object - any tensor shape whose last dimension is of size two [lower_bound, upper_bound]
    :return: tensor of boolean values of the shape of <arr>
    """
    limit_shape = limits.dimensions[:-1]
    arr_shape = arr.dimensions
    assert limit_shape == arr_shape, "Can't use zip_is_in_limits since the shapes of <arr> (%s) " \
                             "and <limits> (%s) do not agree" % (arr.dimensions, limits.dimensions)
    return np.logical_and(arr >= limits[..., LIMIT_MIN], arr <= limits[..., LIMIT_MAX])

@staticmethod
cdef np.ndarray[np.uint8_t] is_almost_in_limits(np.ndarray[np.float_t] arr, list limits):
    """
    tests if values of arr are in the limit [lb, ub] or very close to the limits
    :param arr: any tensor shape
    :param limits: Limits object - 1D numpy array of [lower_bound, upper_bound]
    :return: tensor of boolean values of the shape of <arr>
    """
    return np.logical_or(np.logical_and(arr >= limits[LIMIT_MIN], arr <= limits[LIMIT_MAX]),
                         np.logical_or(np.isclose(arr, limits[LIMIT_MIN]), np.isclose(arr, limits[LIMIT_MAX])))

@staticmethod
cdef np.ndarray[np.float_t] div(np.ndarray[np.float_t] a, np.ndarray[np.float_t] b):
    """
    simple numpy vision operation with handling of division by zero (in that case returns 0)
    :param a: divided part
    :param b: divisor
    :return: a/b where b!=0, 0 otherwise
    """
    return np.true_divide(a, b, out=np.zeros_like(a, dtype=np.float)+np.zeros_like(b, dtype=np.float), where=b != 0)

#
# class UniformGrid:
#     """Lean version of a uniform grid (inclusive)"""
#     def __init__(self, limits: Limits, resolution: float):
#         self.start = limits[LIMIT_MIN]
#         self.end = limits[LIMIT_MAX]
#         self.resolution = resolution
#         self.length = int((self.end-self.start + np.finfo(np.float32).eps) // resolution) + 1
#
#     @property
#     def array(self):
#         return np.arange(self.start, self.end + np.finfo(np.float32).eps, self.resolution)
#
#     def __str__(self):
#         # NOTE: DO NOT CHANGE THIS METHOD WITHOUT ADAPTING FilterBadExpectedTrajectory.validate_predicate_constants METHOD!!!
#         return 'UniformGrid(%s to %s, resolution: %s, length: %s)' % \
#                (self.start, self.end, self.resolution, self.length)
#
#     def __eq__(self, other):
#         return (self.start == other.start and
#                 self.end == other.end and
#                 self.resolution == other.resolution)
#
#     def __len__(self):
#         return self.length
#
#     def __iter__(self):
#         return self.array.__iter__()
#
#     def __getitem__(self, item):
#         """Returns the value from the grid, by index"""
#         if item >= self.length or item < 0:
#             raise IndexError("Index %s is out of bounds for %s" % (item, self))
#         return self.start + item * self.resolution
#
#     def get_index(self, value):
#         """
#         Returns index of closest value on equally-spaced axis
#         :param value: the value to be looked for on axis
#         :return: index of the closest value on the equally-spaced axis
#         """
#         return self.get_indices(np.array([value]))[0]
#
#     def get_indices(self, values):
#         """
#         Returns index of closest value on equally-spaced axis
#         :param values: the value (or array of values) to be looked for on axis
#         :return: index (or array of indices) of the closest value on the equally-spaced axis
#         """
#         assert np.logical_and(self.start <= values, values <= self.end).all(), "values %s are outside the grid %s" % \
#                                                                                (values, str(self))
#         indices = np.round((values - self.start) / self.resolution)
#         return np.clip(indices, 0, self.length - 1).astype(np.int)

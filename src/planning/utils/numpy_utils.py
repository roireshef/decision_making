import numpy as np
from decision_making.src.planning.types import Limits, LIMIT_MIN, LIMIT_MAX


class NumpyUtils:
    @staticmethod
    def cartesian_product_matrix_rows(mat1: np.ndarray, mat2: np.ndarray):
        """
        this function takes two 2D numpy arrays (treats them as matrices) and builds a new matrix whose rows are the
        cartesian products of the rows of the two original matrices
        :param mat1: 2D numpy array [r1 x c1]
        :param mat2: 2D numpy array [r2 x c2]
        :return: 2D numpy array [r1*r2 x (c1+c2)]
        """
        mat1_shape_for_tile = np.ones_like(mat1.shape)
        mat1_shape_for_tile[0] = len(mat1)
        return np.concatenate((np.repeat(mat1, len(mat2), axis=0), np.tile(mat2, tuple(mat1_shape_for_tile))),
                              axis=len(mat1.shape) - 1)

    @staticmethod
    def row_wise_normal(mat: np.ndarray):
        """
        Utility function that takes a 2D matrix of shape [N, 2] and computes a normal vector for each row, assuming
         mat[i, 0] and mat[i, 1] are (x_i, y_i) coordinates of a vector.
        :param mat: 2D numpy array with shape [N, 2]
        :return: 2D numpy array with shape [N, 2] of normal vectors
        """
        return np.c_[-mat[:, 1], mat[:, 0]]

    @staticmethod
    def str_log(arr: np.ndarray) -> str:
        """
        format array for log (no newlines)
        :param arr: any array shape
        :return: string
        """
        return np.array_repr(arr).replace('\n', '')

    @staticmethod
    def is_in_limits(arr: np.array, limits: Limits):
        """
        tests if values of arr are in the limit [lb, ub]
        :param arr: any tensor shape
        :param limits: Limits object - 1D numpy array of [lower_bound, upper_bound]
        :return: tensor of boolean values of the shape of <arr>
        """
        return np.logical_and(arr >= limits[LIMIT_MIN], arr <= limits[LIMIT_MAX])

    @staticmethod
    def is_almost_in_limits(arr: np.array, limits: Limits):
        """
        tests if values of arr are in the limit [lb, ub] or very close to the limits
        :param arr: any tensor shape
        :param limits: Limits object - 1D numpy array of [lower_bound, upper_bound]
        :return: tensor of boolean values of the shape of <arr>
        """
        return np.logical_or(np.logical_and(arr >= limits[LIMIT_MIN], arr <= limits[LIMIT_MAX]),
                             np.logical_or(np.isclose(arr, limits[LIMIT_MIN]), np.isclose(arr, limits[LIMIT_MAX])))

    @staticmethod
    def div(a: np.array, b: np.array):
        """
        simple numpy vision operation with handling of division by zero (in that case returns 0)
        :param a: divided part
        :param b: divisor
        :return: a/b where b!=0, 0 otherwise
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.divide(a, b)
            c[np.isinf(c)] = 0
            c = np.nan_to_num(c)

        return c


class UniformGrid:
    """Lean version of a uniform grid (inclusive)"""
    def __init__(self, limits: Limits, resolution: float):
        self.start = limits[LIMIT_MIN]
        self.end = limits[LIMIT_MAX]
        self.resolution = resolution
        self.length = int((self.end-self.start + np.finfo(np.float32).eps) // resolution) + 1

    @property
    def array(self):
        return np.arange(self.start, self.end + np.finfo(np.float32).eps, self.resolution)

    def __str__(self):
        # NOTE: DO NOT CHANGE THIS METHOD WITHOUT ADAPTING FilterBadExpectedTrajectory.validate_predicate_constants METHOD!!!
        return 'UniformGrid(%s to %s, resolution: %s, length: %s)' % \
               (self.start, self.end, self.resolution, self.length)

    def __eq__(self, other):
        return (self.start == other.start and
                self.end == other.end and
                self.resolution == other.resolution)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.array.__iter__()

    def __getitem__(self, item):
        """Returns the value from the grid, by index"""
        if item >= self.length or item < 0:
            raise IndexError("Index %s is out of bounds for %s" % (item, self))
        return self.start + item * self.resolution

    def get_index(self, value):
        """
        Returns index of closest value on equally-spaced axis
        :param value: the value to be looked for on axis
        :return: index of the closest value on the equally-spaced axis
        """
        return self.get_indices(np.array([value]))[0]

    def get_indices(self, values):
        """
        Returns index of closest value on equally-spaced axis
        :param values: the value (or array of values) to be looked for on axis
        :return: index (or array of indices) of the closest value on the equally-spaced axis
        """
        assert np.logical_and(self.start <= values, values <= self.end).all(), "values %s are outside the grid %s" % \
                                                                               (values, str(self))
        indices = np.round((values - self.start) / self.resolution)
        return np.clip(indices, 0, self.length - 1).astype(np.int)

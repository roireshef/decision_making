import numpy as np


class TensorOps:
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
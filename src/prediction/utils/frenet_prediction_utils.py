import numpy as np
from decision_making.src.planning.types import FS_SV, FS_SX, \
    FrenetState1D, \
    FrenetTrajectories1D, FrenetStates2D, FrenetTrajectories2D, FS_DX


class FrenetPredictionUtils:

    @staticmethod
    def predict_1d_frenet_states(objects_fstates: FrenetState1D, horizons: np.ndarray) -> FrenetTrajectories1D:
        """
        Constant velocity prediction for all timestamps and objects in a matrix computation
        :param objects_fstates: numpy 2D array [Nx3] where N is the number of objects, each row is an FSTATE
        :param horizons: numpy 1D array [T] with T horizons (relative time for prediction into the future)
        :return: numpy 3D array [NxTx3]
        """
        T = horizons.shape[0]
        N = objects_fstates.shape[0]
        if N == 0:
            return []
        zero_slice = np.zeros([N, T])

        s = objects_fstates[:, FS_SX, np.newaxis] + objects_fstates[:, np.newaxis, FS_SV] * horizons
        v = np.tile(objects_fstates[:, np.newaxis, FS_SV], T)

        return np.dstack((s, v, zero_slice))

    @staticmethod
    def predict_2d_frenet_states(objects_fstates: FrenetStates2D, horizons: np.ndarray) -> FrenetTrajectories2D:
        """
        Constant velocity prediction for all timestamps and objects in a matrix computation
        :param objects_fstates: numpy 2D array [Nx6] where N is the number of objects, each row is an FSTATE
        :param horizons: numpy 1D array [T] with T horizons (relative time for prediction into the future)
        :return: numpy 3D array [NxTx6]
        """
        T = horizons.shape[0]
        N = objects_fstates.shape[0]
        if N == 0:
            return []
        zero_slice = np.zeros([N, T])

        s = objects_fstates[:, FS_SX, np.newaxis] + objects_fstates[:, np.newaxis, FS_SV] * horizons
        v = np.tile(objects_fstates[:, np.newaxis, FS_SV], T)
        d = np.tile(objects_fstates[:, np.newaxis, FS_DX], T)

        return np.dstack((s, v, zero_slice, d, zero_slice, zero_slice))

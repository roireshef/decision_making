from dipy.tracking.metrics import frenet_serret

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION, TRAJECTORY_CURVE_INTERP_TYPE
from decision_making.src.planning.types import CartesianPath3D, C_X, C_Y, FP_SX, FP_DX, FrenetPoint, CartesianPoint2D, \
    FrenetTrajectory, CartesianPath2D, FrenetTrajectories, CartesianExtendedTrajectories, FS_SX, FS_SV, FS_SA, FS_DV, \
    FS_DX, FS_DA
from mapping.src.transformations.geometry_utils import CartesianFrame, Euclidean

import numpy as np


class FrenetSerretFrame:
    def __init__(self, points: CartesianPath3D, resolution: float = TRAJECTORY_ARCLEN_RESOLUTION):
        self.O, self._ds = CartesianFrame.resample_curve(curve=points, step_size=resolution / 4,
                                                                  interp_type=TRAJECTORY_CURVE_INTERP_TYPE)
        self.O, self._ds = CartesianFrame.resample_curve(curve=points, step_size=resolution,
                                                                  interp_type='linear')
        self.T, self.N, self.B, self.k, self.t = frenet_serret(self.O)

        self.segments_longitudes = np.linalg.norm(np.diff(self.O, axis=0), axis=1)
        # self.O_cummulative_longitude = np.concatenate([0], np.cumsum(self.segments_longitudes[:-1]))

    def fpoints_to_cpoints(self, fpoints: FrenetTrajectory) -> CartesianPath2D:
        """
        Transforms frenet-frame point to cartesian-frame point (using self.curve) \n
        :param fpoint: numpy array of frenet-point [sx, dx]
        :return: cartesian-frame point [x, y]
        """
        sx, dx = fpoints[:, FP_SX], fpoints[:, FP_DX]
        O_idx, s_leftover = (sx // self._ds).astype(np.int), sx % self._ds

        exact_O = self.O[O_idx, :2] + s_leftover * self.T[O_idx, :2]

        return exact_O + self.N[O_idx, :2] * dx

    def cpoints_to_fpoints(self, cpoints: CartesianPoint2D) -> FrenetTrajectory:
        O_idx, progress = Euclidean.project_on_piecewise_linear_curve(cpoints)
        s_leftover = progress * self.segments_longitudes[O_idx]
        sx = self._ds * O_idx + s_leftover

        exact_O = self.O[O_idx, :2] + s_leftover * self.T[O_idx, :2]

        # this is an approximation in cases where a point lies in the funnel of two segments
        dx = np.linalg.norm(cpoints - exact_O, axis=1)
        return np.dstack((sx, dx))

    def ftrajectories_to_ctrajectories(self, ftrajectories: FrenetTrajectories) -> CartesianExtendedTrajectories:
        """
        Transforms Frenet-frame trajectories to cartesian-frame trajectories, using tensor operations
        :param ftrajectories: Frenet-frame trajectories (tensor)
        :return: Cartesian-frame trajectories (tensor)
        """
        num_t = ftrajectories.shape[0]
        num_p = ftrajectories.shape[1]

        s_x = ftrajectories[:, :, FS_SX]
        s_v = ftrajectories[:, :, FS_SV]
        s_a = ftrajectories[:, :, FS_SA]
        d_x = ftrajectories[:, :, FS_DX]
        d_v = ftrajectories[:, :, FS_DV]
        d_a = ftrajectories[:, :, FS_DA]
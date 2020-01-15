import numpy as np
from decision_making.src.global_constants import FILTER_V_T_GRID, FILTER_V_0_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils


class BrakingDistances:
    """
    Calculates braking distances
    """
    @staticmethod
    def create_braking_distances(aggressiveness_level: AggressivenessLevel) -> np.array:
        """
        Creates distances of all follow_lane with the given aggressiveness_level.
        :return: the actions' distances
        """
        # create v0 & vT arrays for all braking actions
        v0, vT = np.meshgrid(FILTER_V_0_GRID.array, FILTER_V_T_GRID.array, indexing='ij')
        v0, vT = np.ravel(v0), np.ravel(vT)
        # calculate distances for braking actions
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness_level.value]
        distances = np.zeros_like(v0)
        distances[v0 > vT], _ = KinematicUtils.specify_quartic_actions(w_T, w_J, v0[v0 > vT], vT[v0 > vT])
        return distances.reshape(len(FILTER_V_0_GRID), len(FILTER_V_T_GRID))

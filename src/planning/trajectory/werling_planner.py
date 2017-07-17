from src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from src.planning.utils.geometry_utils import FrenetMovingFrame
from src.planning.utils.columns import *
import numpy as np

class WerlingPlanner(TrajectoryPlanner):
    def plan(self, state, reference_route, goal, cost_params):
        frenet = FrenetMovingFrame(reference_route)
        route = frenet.curve

        # TODO: replace this object
        ego = state.ego_state

        fego = frenet.cpoint_to_fpoint([0, 0])      # the ego-vehicle origin in the road-frenet-frame
        ego_theta_diff = route[0, C_THETA]

        # TODO: fix velocity jitters at the State level
        ego_v_x = np.max(ego.v_x, 0.0)

        pass
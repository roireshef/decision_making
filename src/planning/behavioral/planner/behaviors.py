import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import numpy as np

from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner
from decision_making.src.state.map_state import MapState
from decision_making.src.planning import types
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, QuinticPoly1D
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.messages.trajectory_parameters import TrajectoryParams, TrajectoryCostParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.behavioral.planner import blackboard


class VerifyStraightOneWay(Behaviour):
    """
    Verifies we're currently in straight one way road
    """
    def __init__(self, name="verify_straight_one_way"):
        super(VerifyStraightOneWay, self).__init__(name=name)

    def update(self):
        """
        TODO: Verify currently driven road is straight one way
        :return:
        """
        self.feedback_message = "OneWay Straight"
        return Status.SUCCESS


class FollowNominalPath(Behaviour):
    """
    Follows nominal path
    """
    def __init__(self, name="follow_nominal_path"):
        super(FollowNominalPath, self).__init__(name=name)

    def update(self):
        """

        :return:
        """
        ego_lane_id = blackboard.ego_lane_id
        ego_fstate = blackboard.ego_fstate
        nominal_path = blackboard.nominal_path
        state = blackboard.state

        # get desired terminal velocity
        v_T = MapUtils.get_lane(ego_lane_id).e_v_nominal_speed

        # T - terminal time, currently just 10 secs
        T = 10.

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuarticPoly1D.distance_profile_function(a_0=ego_fstate[:, types.FS_SA],
                                                             v_0=ego_fstate[:, types.FS_SV],
                                                             v_T=v_T,
                                                             T=T)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + ego_fstate[:, types.FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        goal_fstate = np.array([target_s, v_T, 0, 0., 0, 0])

        # calculate trajectory cost_params using original goal map_state (from the map)
        goal_segment_id, goal_segment_fstate = nominal_path.convert_to_segment_state(goal_fstate)
        cost_params = CostBasedBehavioralPlanner._generate_cost_params(map_state=MapState(goal_segment_fstate, goal_segment_id),
                                                                       ego_size=state.ego_state.size)  # type: TrajectoryCostParams
        # Calculate cartesian coordinates of action_spec's target (according to target-lane frenet_frame)
        goal_cstate = nominal_path.fstate_to_cstate(goal_fstate)

        # create TrajectoryParams for TP
        trajectory_parameters = TrajectoryParams(reference_route=nominal_path,
                                                 time=state.ego_state.timestamp_in_sec + T,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY,
                                                 bp_time=state.ego_state.timestamp)

        visualization_message = BehavioralVisualizationMsg(
            reference_route_points=trajectory_parameters.reference_route.points)

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

        constraints_s = np.concatenate((ego_fstate[types.FS_SX:(types.FS_SA + 1)],
                                        goal_fstate[types.FS_SX:(types.FS_SA + 1)]))
        constraints_d = np.concatenate((ego_fstate[types.FS_DX:(types.FS_DA + 1)],
                                        goal_fstate[types.FS_DX:(types.FS_DA + 1)]))

        poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
        poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        baseline_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=state.ego_state.timestamp,
                                          T_s=T,
                                          T_d=T,
                                          frenet_frame=nominal_path,
                                          poly_s_coefs=poly_coefs_s,
                                          poly_d_coefs=poly_coefs_d)

        blackboard.tp_input = trajectory_parameters, baseline_trajectory, visualization_message


def straight_one_way_generator():
    verifier = VerifyStraightOneWay()
    action = FollowNominalPath()
    return py_trees.composites.Sequence(name="straight_one_way", children=[verifier, action])

import py_trees

from logging import Logger
from decision_making.src.planning.behavioral.planner.behavioral_planner import BehavioralPlannerBase
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State
from decision_making.src.planning import types
from decision_making.src import global_constants
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.behavioral.planner import blackboard


class BehaviorTreePlanner(BehavioralPlannerBase):
    def __init__(self, tree_generator: callable, predictor: EgoAwarePredictor, logger: Logger):
        super().__init__(predictor, logger)
        self.tree = py_trees.trees.BehaviourTree(tree_generator())

    def _prepare_data(self, state: State, nav_plan: NavigationPlanMsg):
        # get ego frenet state from state data
        ego_fstate = state.ego_state.map_state.lane_fstate
        ego_lane_id = state.ego_state.map_state.lane_id

        # extract nominal path
        nominal_path = MapUtils.get_lookahead_frenet_frame(lane_id=ego_lane_id,
                                                           starting_lon=ego_fstate[types.FS_SX],
                                                           lookahead_dist=global_constants.MAX_HORIZON_DISTANCE,
                                                           navigation_plan=nav_plan)  # type: GeneralizedFrenetSerretFrame
        blackboard.ego_fstate = ego_fstate
        blackboard.ego_lane_id = ego_lane_id
        blackboard.nominal_path = nominal_path
        blackboard.state = state
        blackboard.nav_plan = nav_plan

    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        self._prepare_data(state, nav_plan)
        self.tree.tick()
        trajectory_parameters, baseline_trajectory, visualization_message = blackboard.tp_input
        return trajectory_parameters, baseline_trajectory, visualization_message



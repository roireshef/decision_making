import numpy as np
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.road_sign_action_space import RoadSignActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING, DEFAULT_ROAD_SIGN_RECIPE_FILTERING
from decision_making.src.planning.behavioral.planner.lane_change_planner import LaneChangePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from rte.python.logger.AV_logger import AV_Logger


def test_createSafeActions():
    logger = AV_Logger.get_logger("test")
    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING, 30),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING),
                                                 RoadSignActionSpace(logger, predictor, DEFAULT_ROAD_SIGN_RECIPE_FILTERING)])

    ego_fstate = np.array([100, 0, -0.802153574])
    ego_len = 5
    actor1 = LaneMergeActorState(-146.02568054, 22.222, ego_len)
    actor2 = LaneMergeActorState(-75.05854034, 22.222, ego_len)
    actor3 = LaneMergeActorState(-37.43302536, 22.222, ego_len)
    actors = [actor1, actor2, actor3]
    merge_from_s = 0
    red_line_s = np.inf
    front_actor = LaneMergeActorState(135.340492248535156, 0, ego_len)
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, front_actor, merge_from_s, red_line_s)

    margin = LaneChangePlanner.calculate_margin_to_keep_from_front(state)

    planner = LaneChangePlanner(logger)
    actions = planner._create_action_specs(state, None)
    costs = planner._evaluate_actions(state, None, actions)
    best_idx = np.argmin(costs)
    print('best_idx=', best_idx)
    [print(spec) for spec in actions[best_idx].action_specs]


def test_calc_margin():
    margin = LaneChangePlanner.calc_headway_margin(target_v=20, front_v=8, v_S=0)
    margin = margin
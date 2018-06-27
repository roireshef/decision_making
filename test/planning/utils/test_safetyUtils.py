from logging import Logger
import time
import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.behavioral.evaluators.test_heuristicActionSpecEvaluator import create_canonic_ego, \
    get_road_rhs_frenet_by_road_id, create_canonic_object
from decision_making.src.state.state import ObjectSize, State
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone


def test_calcSafety():
    logger = Logger("test_SafetyUtils")
    lane_width = 3.6
    ego_size = ObjectSize(4, 2, 0)
    ego_lon = 400
    ego_lat = lane_width / 2
    ego_vel = 10
    road_id = 20
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)

    t = 20.
    times_step = 0.2
    time_samples = np.arange(0, t + 0.001, times_step)
    samples_num = time_samples.shape[0]

    ego = create_canonic_ego(0, ego_lon, ego_lat, ego_vel, ego_size, road_frenet)
    ego_fstate = np.array([ego.road_localization.road_lon, ego.v_x, 0, ego.road_localization.intra_road_lat, 0, 0])

    obj_sizes = np.array([ObjectSize(4, 2, 0), ObjectSize(6, 2, 0), ObjectSize(6, 2, 0), ObjectSize(4, 2, 0)])

    F = create_canonic_object(1, 0, ego_lon + 50, ego_lat, ego_vel, obj_sizes[0], road_frenet)
    LF = create_canonic_object(2, 0, ego_lon + 20, ego_lat + lane_width, ego_vel + 6, obj_sizes[1], road_frenet)
    # L = create_canonic_object(3, 0, ego_lon - 3, ego_lat + lane_width, ego_vel, obj_sizes[2], road_frenet)
    LB = create_canonic_object(4, 0, ego_lon - 40, ego_lat + lane_width, ego_vel + 2, obj_sizes[3], road_frenet)
    objects = [F, LF, LB]

    predictions = {}
    for i, obj in enumerate(objects):
        fstate = np.array([obj.road_localization.road_lon, obj.v_x, 0, obj.road_localization.intra_road_lat, 0, 0])
        prediction = np.tile(fstate, samples_num).reshape(samples_num, 6)
        prediction[:, 0] = fstate[FS_SX] + time_samples * fstate[FS_SV]
        predictions[obj.obj_id] = prediction

    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

    # Action specification
    specs = np.full(recipes.__len__(), None)
    valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
    specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

    specs_mask = action_spec_validator.filter_action_specs(list(specs), behavioral_state)

    st = time.time()
    intervals = SafetyUtils.calc_safety(behavioral_state, ego_fstate, list(recipes), list(specs), specs_mask,
                                        predictions, time_samples)
    print('total time=%f' % (time.time()-st))

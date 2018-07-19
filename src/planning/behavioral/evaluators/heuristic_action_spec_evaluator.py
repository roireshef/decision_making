from logging import Logger
from typing import List, Optional

import numpy as np
import copy
import time
import sys

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, SAFETY_MARGIN_TIME_DELAY, \
    LAT_CALM_ACC, MINIMAL_STATIC_ACTION_TIME, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, \
    RelativeLane, ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_CostFunctions
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.types import FP_SX, FS_DV, FS_DX, FS_SX, FS_SV, FrenetState2D, FS_DA, LIMIT_MAX
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from mapping.src.service.map_service import MapService


class HeuristicActionSpecEvaluator(ActionSpecEvaluator):
    """
    Link to the algorithm documentation in confluence:
    https://confluence.gm.com/display/SHAREGPDIT/BP+costs+and+heuristic+assumptions
    """
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor):
        super().__init__(logger)
        self.predictor = predictor

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
        """
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
        all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_recipes: array of actions recipes
        :param action_specs: array of action specs
        :param action_specs_mask: array of boolean values: mask[i]=True if specs[i] was not filtered
        :return: array of costs (one cost per action)
        """
        ego = behavioral_state.ego_state
        lane_width = MapService.get_instance().get_road(ego.map_state.road_id).lane_width
        ego_fstate = ego.map_state.road_fstate
        ego_lane = int(ego_fstate[FS_DX] / lane_width)

        print('\ntime=%.1f ego_lon=%.2f ego_v=%.2f ego_lat=%.2f ego_dv=%.2f grid_size=%d' %
              (ego.timestamp_in_sec, ego_fstate[FS_SX], ego.velocity, ego_fstate[FS_DX],
               ego_fstate[FS_DV], len(behavioral_state.road_occupancy_grid)))

        costs = np.full(len(action_recipes), np.inf)
        specs = copy.deepcopy(action_specs)

        times_step = 0.1
        time_samples = np.arange(0, BP_ACTION_T_LIMITS[LIMIT_MAX], times_step)

        # TODO: use fast predictor
        grid = behavioral_state.road_occupancy_grid
        obj_ids = np.array([grid[cell][0].dynamic_object.obj_id for cell in grid])
        objects_curr_fstates = np.array([grid[cell][0].dynamic_object.map_state.road_fstate for cell in grid])
        predicted_fstates = self.predictor.predict_frenet_states(objects_curr_fstates, time_samples)
        predictions = {}
        for i, obj_id in enumerate(obj_ids):
            predictions[obj_id] = predicted_fstates[i]

        st = time.time()

        all_safe_intervals = SafetyUtils.calc_safety(behavioral_state, ego_fstate, action_recipes, specs, action_specs_mask,
                                                     predictions, time_samples)

        print('calc_safety time = %f' % (time.time()-st))

        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if action_specs_mask[i]])
        spec_orig_idxs = specs_arr[:, 0].astype(int)
        specs_t = specs_arr[:, 1]
        specs_v = specs_arr[:, 2]
        specs_s = specs_arr[:, 3]
        specs_d = specs_arr[:, 4]
        # calculate approximated lateral time according to the CALM aggressiveness level
        T_d_array = HeuristicActionSpecEvaluator._calc_lateral_times(ego_fstate, specs_t, specs_d)

        # loop over all specs / actions
        for i, spec_arr in enumerate(specs_arr):
            if not action_specs_mask[spec_orig_idxs[i]]:
                continue

            recipe = action_recipes[spec_orig_idxs[i]]
            spec = specs[spec_orig_idxs[i]]

            T_d = T_d_array[i]

            safe_intervals = []
            if len(all_safe_intervals) > 0:
                safe_intervals = all_safe_intervals[np.where(all_safe_intervals[:, 0] == spec_orig_idxs[i])][:, 1:]

            if len(safe_intervals) == 0 or safe_intervals[0, 0] > 0:
                print('unsafe action %3d(%d): lane %d dist=%.2f [t=%.2f td=%.2f s=%.2f v=%.2f]' %
                      (spec_orig_idxs[i], recipe.aggressiveness.value, ego_lane + recipe.relative_lane.value,
                       HeuristicActionSpecEvaluator._dist_to_target(behavioral_state, ego_fstate, spec),
                       spec.t, T_d, spec.s - ego_fstate[0], spec.v))
                continue
            T_d_max = safe_intervals[0, 1]

            # calculate actions costs
            sub_costs = HeuristicActionSpecEvaluator._calc_action_costs(ego_fstate, spec, lane_width, T_d_max, T_d)
            costs[spec_orig_idxs[i]] = np.sum(sub_costs)

            print('action %d(%d %d) lane %d: dist=%.1f [t=%.2f td=%.2f tdmax=%.2f s=%.2f v=%.2f] '
                  '[eff %.3f comf %.2f,%.2f right %.2f dev %.2f]: tot %.2f' %
                  (spec_orig_idxs[i], recipe.action_type.value, recipe.aggressiveness.value, ego_lane + recipe.relative_lane.value,
                   HeuristicActionSpecEvaluator._dist_to_target(behavioral_state, ego_fstate, spec),
                   spec.t, T_d, T_d_max, spec.s - ego_fstate[0], spec.v,
                   sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], costs[spec_orig_idxs[i]]))

            self.logger.debug("action %d(%d %d) lane %d: dist=%.1f [t=%.2f td=%.2f s=%.2f v=%.2f] "
                              "[eff %.3f comf %.2f,%.2f right %.2f dev %.2f]: tot %.2f",
                              spec_orig_idxs[i], recipe.action_type.value, recipe.aggressiveness.value,
                              ego_lane + recipe.relative_lane.value,
                              HeuristicActionSpecEvaluator._dist_to_target(behavioral_state, ego_fstate, spec),
                              spec.t, T_d, spec.s - ego_fstate[0], spec.v,
                              sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], costs[spec_orig_idxs[i]])

        if np.isinf(np.min(costs)):
            print("********************  NO SAFE ACTION!  **********************")
            self.logger.warning("********************  NO SAFE ACTION!  **********************")
        else:
            best_action = int(np.argmin(costs))
            self.logger.debug("Best action %d; lane %d\n", best_action,
                              ego_lane + action_recipes[best_action].relative_lane.value)

        # print('time: t1=%f t2=%f t3=%f' % (t1, t2, t3))

        return costs

    # @staticmethod
    # def _calc_velocity_profile(ego_fstate: np.array, recipe: ActionRecipe, spec: ActionSpec) -> VelocityProfile:
    #     """
    #     Given action recipe and behavioral state, calculate the longitudinal velocity profile for that action.
    #     :param ego_fstate: current ego Frenet state
    #     :param recipe: the input action
    #     :param spec: action specification
    #     :return: longitudinal velocity profile or None if the action is infeasible by given aggressiveness level
    #     """
    #     if recipe.action_type == ActionType.FOLLOW_VEHICLE or recipe.action_type == ActionType.OVERTAKE_VEHICLE:
    #         dist = spec.s - spec.t * spec.v - ego_fstate[FS_SX]
    #         return VelocityProfile.calc_profile_given_T(ego_fstate[FS_SV], spec.t, dist, spec.v)
    #     else:  # static action (FOLLOW_LANE)
    #         t2 = max(0., MINIMAL_STATIC_ACTION_TIME - spec.t)  # TODO: remove it after implementation of value function
    #         return VelocityProfile(v_init=ego_fstate[FS_SV], t_first=spec.t, v_mid=spec.v, t_flat=t2, t_last=0, v_tar=spec.v)

    @staticmethod
    def _calc_lateral_times(ego_fstate: FrenetState2D, specs_t: np.array, specs_d: np.array) -> np.array:
        """
        Given initial lateral velocity and signed lateral distance, estimate a time it takes to perform the movement.
        The time estimation assumes movement by velocity profile like in the longitudinal case.
        :param ego_fstate: initial ego frenet state
        :param specs_t: array of time specifications
        :param specs_d: array of lateral distances of action specifications
        :return: [s] the lateral movement time to the target, [m] maximal lateral deviation from lane center,
        [m/s] initial lateral velocity toward target (negative if opposite to the target direction)
        """
        calm_weights = np.array([1.5, 1])  # calm lateral movement
        T_d = HeuristicActionSpecEvaluator._calc_T_d(calm_weights, ego_fstate, specs_d)
        return np.minimum(T_d, specs_t)

    @staticmethod
    def _calc_T_d(weights: np.array, ego_init_fstate: FrenetState2D, specs_d: np.array) -> np.array:
        """
        Calculate lateral movement time for the given Jerk/T weights.
        :param weights: array of size 2: weights[0] is jerk weight, weights[1] is T weight
        :param ego_init_fstate: ego initial frenet state
        :param specs_d: array of lateral distances of action specifications
        :return: array of lateral movement times
        """
        specs_num = len(specs_d)
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=np.repeat(weights[1], specs_num), w_J=np.repeat(weights[0], specs_num), dx=specs_d - ego_init_fstate[FS_DX],
            a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV], v_T=0, T_m=0)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)
        return T_d

    @staticmethod
    def _calc_action_costs(ego_fstate: np.array, spec: ActionSpec, lane_width: float,
                           T_d_max: float, T_d_approx: float) -> [float, np.array]:
        """
        Calculate the cost of the action
        :param spec: action spec
        :param lane_width: lane width
        :param T_d_max: [sec] the largest possible lateral time imposed by safety. np.inf if it's not imposed
        :param T_d_approx: [sec] heuristic approximation of lateral time, according to the initial and end constraints
        :return: the action's cost and the cost components array (for debugging)
        """
        # calculate efficiency, comfort and non-right lane costs
        target_lane = int(spec.d / lane_width)
        efficiency_cost = BP_CostFunctions.calc_efficiency_cost(ego_fstate, spec)
        lon_comf_cost, lat_comf_cost = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, T_d_max, T_d_approx)
        right_lane_cost = BP_CostFunctions.calc_right_lane_cost(spec.t, target_lane)

        # calculate maximal deviation from lane center for lane deviation cost
        signed_lat_dist = spec.d - ego_fstate[FS_DX]
        rel_lat = abs(signed_lat_dist)/lane_width
        rel_vel = ego_fstate[FS_DV]/lane_width
        if signed_lat_dist * rel_vel < 0:  # changes lateral direction
            rel_lat += rel_vel*rel_vel/(2*LAT_CALM_ACC)  # predict maximal deviation
        max_lane_dev = min(2*rel_lat, 1)  # for half-lane deviation, max_lane_dev = 1
        lane_deviation_cost = BP_CostFunctions.calc_lane_deviation_cost(max_lane_dev)
        return np.array([efficiency_cost, lon_comf_cost, lat_comf_cost, right_lane_cost, lane_deviation_cost])

    @staticmethod
    def _dist_to_target(state: BehavioralGridState, ego_fstate: FrenetState2D, spec: ActionSpec):
        """
        given action recipe, calculate longitudinal distance from the target object, and inf for static action
        :param state: behavioral state
        :param spec: action specification
        :return: distance from the target
        """
        lane_width = MapService.get_instance().get_road(state.ego_state.map_state.road_id).lane_width
        _, rel_lanes = SafetyUtils._get_rel_lane_from_specs(lane_width, ego_fstate, np.array([spec.d]))
        forward_cell = (rel_lanes[0], RelativeLongitudinalPosition.FRONT)
        dist = np.inf
        if forward_cell in state.road_occupancy_grid:
            cell_fstate = state.road_occupancy_grid[forward_cell][0].dynamic_object.map_state.road_fstate
            dist = cell_fstate[FS_SX] - ego_fstate[FS_SX]
        return dist

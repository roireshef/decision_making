from logging import Logger
from typing import List, Optional

import numpy as np

from decision_making.src.exceptions import BehavioralPlanningException, InvalidAction
from decision_making.src.exceptions import NoValidTrajectoriesFound, raises
from decision_making.src.global_constants import EGO_ORIGIN_LON_FROM_REAR, TRAJECTORY_ARCLEN_RESOLUTION, \
    PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, VELOCITY_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, SEMANTIC_CELL_LON_REAR, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.global_constants import OBSTACLE_SIGMOID_COST, \
    DEVIATION_FROM_ROAD_COST, DEVIATION_TO_SHOULDER_COST, \
    DEVIATION_FROM_LANE_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_K_PARAM, \
    DEVIATION_FROM_GOAL_COST, DEVIATION_FROM_GOAL_LAT_LON_RATIO, GOAL_SIGMOID_K_PARAM, \
    GOAL_SIGMOID_OFFSET, LATERAL_SAFETY_MARGIN_FROM_OBJECT, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, SHOULDER_SIGMOID_OFFSET, LON_JERK_COST, LAT_JERK_COST, LANE_SIGMOID_K_PARAM, \
    SHOULDER_SIGMOID_K_PARAM, BP_ACTION_T_LIMITS, \
    BP_ACTION_T_RES, SAFE_DIST_TIME_DELAY, SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_RIGHT, MIN_OVERTAKE_VEL
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import \
    SemanticActionsGridState
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticActionSpec
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticAction, SemanticActionType, \
    LAT_CELL, LON_CELL, SemanticGridCell
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuinticPoly1D, QuarticPoly1D
from decision_making.src.planning.trajectory.optimal_control.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import FS_SA, FS_SV, FS_SX, FS_DX, FS_DV, FS_DA, FP_SX, FrenetPoint
from decision_making.src.planning.types import LIMIT_MIN, LIMIT_MAX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class SemanticActionsGridPolicy(SemanticActionsPolicy):
    def __init__(self, logger: Logger, predictor: Predictor):
        super().__init__(logger=logger, predictor=predictor)
        self._last_ego_state: Optional[EgoState] = None
        self._last_action: Optional[SemanticAction] = None
        self._last_action_spec: Optional[SemanticActionSpec] = None
        self._last_poly_coefs_s: Optional[np.ndarray] = None

    def plan(self, state: State, nav_plan: NavigationPlanMsg):

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = SemanticActionsGridState.create_from_state(state=state,
                                                                      logger=self.logger)

        # # debug: computing distance from other objects, good only for one vehicle in scenario
        # minimal_distance = 1000
        # ego_x = state.ego_state.x
        # ego_y = state.ego_state.y
        # closest_dyn_obj = None
        # for dyn_obj in state.dynamic_objects:
        #     dist = np.sqrt(np.power(dyn_obj.x - ego_x, 2) + np.power(dyn_obj.y - ego_y, 2))
        #     if dist < minimal_distance:
        #         minimal_distance = dist
        #         closest_dyn_obj = dyn_obj
        # print("Ego/object velocities: " + str(state.ego_state.total_speed) + "/" + str(
        #     closest_dyn_obj.total_speed) + " x: " + str(closest_dyn_obj.x) + ", y: " + str(
        #     closest_dyn_obj.y) + ", min dist from obj is: " + str(minimal_distance))

        # iterate over the semantic grid and enumerate all relevant HL actions
        semantic_actions = self._enumerate_actions(behavioral_state=behavioral_state)

        # iterate over all HL actions and generate a specification (desired terminal: position, velocity, time-horizon)
        action_specs = []
        for semantic_action in semantic_actions:
            try:
                action_spec = self._specify_action(behavioral_state=behavioral_state,
                                                   semantic_action=semantic_action,
                                                   navigation_plan=nav_plan)
                action_specs.append(action_spec)
            except InvalidAction as e:
                self.logger.warning(str(e) + " SemanticAction: " + str(semantic_action))
                action_specs.append(None)

        # Filter actions with invalid spec
        valid_spec_indices = [x for x in range(len(action_specs)) if action_specs[x] is not None]
        semantic_actions = [semantic_actions[x] for x in valid_spec_indices]
        actions_spec = [action_specs[x] for x in valid_spec_indices]

        # evaluate all action-specifications by computing a cost for each action
        action_costs = self._eval_actions(behavioral_state=behavioral_state, semantic_actions=semantic_actions,
                                          actions_spec=actions_spec)

        # select an action-specification with minimal cost
        selected_action_index = int(np.argmin(action_costs))
        selected_action_spec = actions_spec[selected_action_index]

        trajectory_parameters = SemanticActionsGridPolicy._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                     action_spec=selected_action_spec,
                                                                                     navigation_plan=nav_plan)

        visualization_message = BehavioralVisualizationMsg(reference_route=trajectory_parameters.reference_route)

        # updating selected actions in memory
        self._last_action = semantic_actions[selected_action_index]
        self._last_action_spec = selected_action_spec
        self._last_ego_state = state.ego_state

        self.logger.debug("Chosen behavioral semantic action is %s, %s",
                          semantic_actions[selected_action_index].__dict__, selected_action_spec.__dict__)

        return trajectory_parameters, visualization_message

    def _enumerate_actions(self, behavioral_state: SemanticActionsGridState) -> List[SemanticAction]:
        """
        Enumerate the list of possible semantic actions to be generated.
        Every cell is being tested for the existence of cars.
        If one or more cars exist in it, we generate an action towards the first object in the list
        (the first object currently refers to the closest one to ego, depending on the semantic grid implementation
        which maps the list of DynamicObject to their respective cells)
        :param behavioral_state: behavioral_state contains semantic_occupancy_grid and ego_state
        :return: the list of semantic actions
        """

        semantic_actions: List[SemanticAction] = list()

        # Go over all cells in road semantic occupancy grid. The grid contains only relevant cells towards which
        # we can plan a trajectory.
        for semantic_cell in behavioral_state.road_occupancy_grid:
            # Generate actions towards each of the cells in front of ego
            if semantic_cell[LON_CELL] == SEMANTIC_CELL_LON_FRONT:
                if len(behavioral_state.road_occupancy_grid[semantic_cell]) > 0:
                    # Select first (closest) object in cell
                    semantic_action = SemanticAction(cell=semantic_cell,
                                                     target_obj=behavioral_state.road_occupancy_grid[semantic_cell][0],
                                                     action_type=SemanticActionType.FOLLOW_VEHICLE)
                else:
                    # There are no objects in cell
                    semantic_action = SemanticAction(cell=semantic_cell,
                                                     target_obj=None,
                                                     action_type=SemanticActionType.FOLLOW_LANE)

                semantic_actions.append(semantic_action)

        return semantic_actions

    # TODO: modify this function to work with DynamicObject's specific NavigationPlan (and predictor?)
    @raises(InvalidAction)
    def _specify_action(self, behavioral_state: SemanticActionsGridState, semantic_action: SemanticAction,
                        navigation_plan: NavigationPlanMsg) -> Optional[SemanticActionSpec]:
        """
        given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param behavioral_state: semantic actions grid behavioral state
        :param semantic_action:
        :return: SemanticActionSpec
        """
        ego = behavioral_state.ego_state

        # BP IF - if ego is close to last planned trajectory (in BP), then assume ego is exactly on this trajectory
        if self._last_action is not None and semantic_action == self._last_action \
                and LocalizationUtils.is_actual_state_close_to_expected_state(
                    ego, self._last_action_spec.samplable_trajectory, self.logger, self.__class__.__name__):
            ego_init_cstate = self._last_action_spec.samplable_trajectory.sample(np.array([ego.timestamp_in_sec]))[0]
        else:
            ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])

        road_id = ego.road_localization.road_id

        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)

        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        if semantic_action.action_type == SemanticActionType.FOLLOW_VEHICLE:
            # part of ego from its origin to its front + half of target object
            lon_margin = ego.size.length / 2 + semantic_action.target_obj.size.length / 2 + \
                         LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

            # TODO: the relative localization calculated here assumes that all objects are located on the same road and Frenet frame.
            # TODO: Fix after demo and calculate longitudinal difference properly in the general case
            return self._specify_follow_vehicle_action(semantic_action.target_obj, road_frenet, ego_init_fstate,
                                                       ego.timestamp_in_sec, lon_margin)

        elif semantic_action.action_type == SemanticActionType.FOLLOW_LANE:
            road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
            desired_lane = ego.road_localization.lane_num + semantic_action.cell[LAT_CELL]
            desired_center_lane_latitude = road_lane_latitudes[desired_lane]

            return self._specify_follow_lane_action(road_frenet, ego_init_fstate, ego.timestamp_in_sec,
                                                    desired_center_lane_latitude)

    @raises(InvalidAction)
    def _specify_follow_lane_action(self, road_frenet: FrenetSerret2DFrame,
                                    ego_init_fstate: np.ndarray, ego_timestamp_in_sec: float,
                                    desired_latitude: float) -> SemanticActionSpec:
        """
        This method's purpose is to specify the enumerated actions that the agent can take.
        Each semantic action is translated to a trajectory of the agent.
        The trajectory specification is created towards a target location/object in given cell,
         considering ego speed, location.
         Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it
        :return: semantic action specification
        """
        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX], BP_ACTION_T_RES)

        # Quartic polynomial constraints (no constraint on sT)
        constraints_s = np.repeat([[
            ego_init_fstate[FS_SX],
            ego_init_fstate[FS_SV],
            ego_init_fstate[FS_SA],
            BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED,  # desired velocity # TODO: change to the road's target speed
            0.0  # zero acceleration at the end of action
        ]], repeats=len(T_vals), axis=0)

        A_inv_s = np.linalg.inv(QuarticPoly1D.time_constraints_tensor(T_vals))
        poly_coefs_s = QuarticPoly1D.zip_solve(A_inv_s, constraints_s)
        target_s = Math.zip_polyval2d(poly_coefs_s, T_vals[:, np.newaxis])

        # Quintic polynomial constraints
        constraints_d = np.repeat([[
            ego_init_fstate[FS_DX],
            ego_init_fstate[FS_DV],
            ego_init_fstate[FS_DA],
            desired_latitude,  # Target latitude relative to reference route (RHS of road)
            0.0,
            0.0
        ]], repeats=len(T_vals), axis=0)

        A_inv_d = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, constraints_d)

        # TODO: acceleration is computed in frenet frame and not cartesian. if road is curved, this is problematic
        are_lon_acc_in_limits = QuarticPoly1D.are_accelerations_in_limits(poly_coefs_s, T_vals, LON_ACC_LIMITS)
        are_vel_in_limits = QuarticPoly1D.are_velocities_in_limits(poly_coefs_s, T_vals, VELOCITY_LIMITS)
        are_lat_acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs_d, T_vals, LAT_ACC_LIMITS)

        jerk_s = QuarticPoly1D.cumulative_jerk(poly_coefs_s, T_vals)
        jerk_d = QuinticPoly1D.cumulative_jerk(poly_coefs_d, T_vals)

        cost = np.dot(np.c_[jerk_s, jerk_d, T_vals], np.c_[BP_JERK_S_JERK_D_TIME_WEIGHTS])
        optimum_time_idx = np.argmin(cost)

        optimum_time_satisfies_constraints = are_lon_acc_in_limits[optimum_time_idx] & \
                              are_lat_acc_in_limits[optimum_time_idx] & \
                              are_vel_in_limits[optimum_time_idx]

        if not optimum_time_satisfies_constraints:
            raise InvalidAction("Couldn't specify action due to unsatisfied constraints. "
                                "Last action spec: %s. Optimal time: %f. Velocity in limits: %s. "
                                "Longitudinal acceleration in limits: %s. Latitudinal acceleration in limits: %s." %
                                  (str(self._last_action_spec), T_vals[optimum_time_idx],
                                   are_vel_in_limits[optimum_time_idx],
                                   are_lon_acc_in_limits[optimum_time_idx],
                                   are_lat_acc_in_limits[optimum_time_idx]))

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego_timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        return SemanticActionSpec(t=T_vals[optimum_time_idx], v=constraints_s[optimum_time_idx, 3],
                                  s=target_s[optimum_time_idx, 0],
                                  d=constraints_d[optimum_time_idx, 3],
                                  samplable_trajectory=samplable_trajectory)

    @raises(InvalidAction)
    def _specify_follow_vehicle_action(self, target_obj: DynamicObject, road_frenet: FrenetSerret2DFrame,
                                       ego_init_fstate: np.ndarray, ego_timestamp_in_sec: float,
                                       lon_margin: float) -> SemanticActionSpec:
        """
        given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param behavioral_state: semantic actions grid behavioral state
        :param semantic_action:
        :return: SemanticActionSpec
        """
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([target_obj.x, target_obj.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            target_obj.x, target_obj.y,
            target_obj.yaw,
            target_obj.total_speed,
            target_obj.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Extract relevant details from state on Reference-Object
        obj_on_road = target_obj.road_localization  # TODO: road localization needs to be handled
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]

        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX], BP_ACTION_T_RES)

        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))

        # TODO: should be swapped with current implementation of Predictor
        obj_saT = 0  # obj_init_fstate[FS_SA]
        obj_svT = obj_init_fstate[FS_SV] + obj_saT * T_vals
        obj_sxT = obj_init_fstate[FS_SX] + obj_svT * T_vals + obj_saT * T_vals ** 2 / 2

        # TODO: account for acc<>0 (from MobilEye's paper)
        safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY

        constraints_s = np.c_[
            np.full(shape=len(T_vals), fill_value=ego_init_fstate[FS_SX]),
            np.full(shape=len(T_vals), fill_value=ego_init_fstate[FS_SV]),
            np.full(shape=len(T_vals), fill_value=ego_init_fstate[FS_SA]),
            obj_sxT - safe_lon_dist - lon_margin,
            obj_svT,
            np.full(shape=len(T_vals), fill_value=obj_saT)
        ]

        constraints_d = np.repeat([[
            ego_init_fstate[FS_DX],
            ego_init_fstate[FS_DV],
            ego_init_fstate[FS_DA],
            obj_center_lane_latitude,
            0.0,
            0.0
        ]], repeats=len(T_vals), axis=0)

        # solve for s(t) and d(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        # TODO: acceleration is computed in frenet frame and not cartesian. if road is curved, this is problematic
        are_lon_acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs_s, T_vals, LON_ACC_LIMITS)
        are_lat_acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs_d, T_vals, LAT_ACC_LIMITS)
        are_vel_in_limits = QuinticPoly1D.are_velocities_in_limits(poly_coefs_s, T_vals, VELOCITY_LIMITS)

        jerk_s = QuinticPoly1D.cumulative_jerk(poly_coefs_s, T_vals)
        jerk_d = QuinticPoly1D.cumulative_jerk(poly_coefs_d, T_vals)

        cost = np.dot(np.c_[jerk_s, jerk_d, T_vals], np.c_[BP_JERK_S_JERK_D_TIME_WEIGHTS])
        optimum_time_idx = np.argmin(cost)

        optimum_time_satisfies_constraints = are_lon_acc_in_limits[optimum_time_idx] & \
                              are_lat_acc_in_limits[optimum_time_idx] & \
                              are_vel_in_limits[optimum_time_idx]

        if not optimum_time_satisfies_constraints:
            raise InvalidAction("Couldn't specify action due to unsatisfied constraints. "
                                "Last action spec: %s. Optimal time: %f. Velocity in limits: %s. "
                                "Longitudinal acceleration in limits: %s. Latitudinal acceleration in limits: %s." %
                                  (str(self._last_action_spec), T_vals[optimum_time_idx],
                                   are_vel_in_limits[optimum_time_idx],
                                   are_lon_acc_in_limits[optimum_time_idx],
                                   are_lat_acc_in_limits[optimum_time_idx]))

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego_timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        return SemanticActionSpec(t=T_vals[optimum_time_idx], v=obj_svT[optimum_time_idx],
                                  s=constraints_s[optimum_time_idx, 3],
                                  d=constraints_d[optimum_time_idx, 3],
                                  samplable_trajectory=samplable_trajectory)

    def _eval_actions(self, behavioral_state: SemanticActionsGridState,
                      semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
        """
        Evaluate the generated actions using the actions' spec and SemanticBehavioralState containing semantic grid.
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
         all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        if len(semantic_actions) != len(actions_spec):
            self.logger.error(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(semantic_actions), len(actions_spec))
            raise BehavioralPlanningException(
                "The input arrays have different sizes: len(semantic_actions)=%d, len(actions_spec)=%d",
                len(semantic_actions), len(actions_spec))

        # get indices of semantic_actions array for 3 actions: goto-right, straight, goto-left
        current_lane_action_ind = SemanticActionsGridPolicy._get_action_ind(
            semantic_actions, (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT))
        left_lane_action_ind = SemanticActionsGridPolicy._get_action_ind(
            semantic_actions, (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_FRONT))
        right_lane_action_ind = SemanticActionsGridPolicy._get_action_ind(
            semantic_actions, (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_FRONT))

        # The cost for each action is assigned so that the preferred policy would be:
        # Go to right if right and current lanes are fast enough.
        # Go to left if the current lane is slow and the left lane is faster than the current.
        # Otherwise remain on the current lane.

        # TODO - this needs to come from map
        desired_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED

        # boolean whether the forward-right cell is fast enough (may be empty grid cell)
        is_forward_right_fast = right_lane_action_ind is not None and \
                                desired_vel - actions_spec[right_lane_action_ind].v < MIN_OVERTAKE_VEL
        # boolean whether the right cell near ego is occupied
        is_right_occupied = True
        if (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_right_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_RIGHT,
                                                                          SEMANTIC_CELL_LON_SAME)]) > 0

        # boolean whether the forward cell is fast enough (may be empty grid cell)
        is_forward_fast = current_lane_action_ind is not None and \
                          desired_vel - actions_spec[current_lane_action_ind].v < MIN_OVERTAKE_VEL

        # boolean whether the forward-left cell is faster than the forward cell
        is_forward_left_faster = left_lane_action_ind is not None and \
                                 (current_lane_action_ind is None or
                                  actions_spec[left_lane_action_ind].v - actions_spec[current_lane_action_ind].v >=
                                  MIN_OVERTAKE_VEL)

        ego = behavioral_state.ego_state
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_fpoint = road_frenet.cpoint_to_fpoint(np.array([ego.x, ego.y]))

        dist_to_backleft, safe_left_dist_behind_ego = SemanticActionsGridPolicy._calc_safe_dist_behind_ego(
            behavioral_state, road_frenet, ego_fpoint, SEMANTIC_CELL_LAT_LEFT)
        dist_to_backright, safe_right_dist_behind_ego = SemanticActionsGridPolicy._calc_safe_dist_behind_ego(
            behavioral_state, road_frenet, ego_fpoint, SEMANTIC_CELL_LAT_RIGHT)

        self.logger.debug("Distance\safe distance to back left car: %s\%s.", dist_to_backleft, safe_left_dist_behind_ego)
        self.logger.debug("Distance\safe distance to back right car: %s\%s.", dist_to_backright, safe_right_dist_behind_ego)

        # boolean whether the left cell near ego is occupied
        if (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_left_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_LEFT,
                                                                         SEMANTIC_CELL_LON_SAME)]) > 0
        else:
            is_left_occupied = True

        costs = np.ones(len(semantic_actions))

        # move right if both straight and right lanes are fast
        # if is_forward_right_fast and (is_forward_fast or current_lane_action_ind is None) and not is_right_occupied:
        if is_forward_right_fast and not is_right_occupied and dist_to_backright > safe_right_dist_behind_ego:
            costs[right_lane_action_ind] = 0.
        # move left if straight is slow and the left is faster than straight
        elif not is_forward_fast and (
                    is_forward_left_faster or current_lane_action_ind is None) and not is_left_occupied and \
                        dist_to_backleft > safe_left_dist_behind_ego:
            costs[left_lane_action_ind] = 0.
        else:
            costs[current_lane_action_ind] = 0.
        return costs

    @staticmethod
    def _calc_safe_dist_behind_ego(behavioral_state: SemanticActionsGridState, road_frenet: FrenetSerret2DFrame,
                                   ego_fpoint: FrenetPoint, semantic_cell_lat: int) -> [float, float]:
        """
        Calculate both actual and safe distances between rear object and ego on the left side or right side.
        If there is no object, return actual dist = inf and safe dist = 0.
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :param road_frenet: road Frenet frame for ego's road_id
        :param ego_fpoint: frenet point of ego location
        :param semantic_cell_lat: either SEMANTIC_CELL_LAT_LEFT or SEMANTIC_CELL_LAT_RIGHT
        :return: longitudinal distance between ego and rear object, safe distance between ego and the rear object
        """
        dist_to_back_obj = np.inf
        safe_dist_behind_ego = 0
        back_objects = []
        if (semantic_cell_lat, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_objects = behavioral_state.road_occupancy_grid[(semantic_cell_lat, SEMANTIC_CELL_LON_REAR)]
        if len(back_objects) > 0:
            back_fpoint = road_frenet.cpoint_to_fpoint(np.array([back_objects[0].x, back_objects[0].y]))
            dist_to_back_obj = ego_fpoint[FP_SX] - back_fpoint[FP_SX]
            if behavioral_state.ego_state.v_x > back_objects[0].v_x:
                safe_dist_behind_ego = back_objects[0].v_x * SAFE_DIST_TIME_DELAY
            else:
                safe_dist_behind_ego = back_objects[0].v_x * SAFE_DIST_TIME_DELAY + \
                        back_objects[0].v_x ** 2 / (2 * abs(LON_ACC_LIMITS[0])) - \
                        behavioral_state.ego_state.v_x ** 2 / (2 * abs(LON_ACC_LIMITS[0]))
        return dist_to_back_obj, safe_dist_behind_ego

    @staticmethod
    def _get_action_ind(semantic_actions: List[SemanticAction], cell: SemanticGridCell):
        """
        Given semantic actions array and action cell, return index of action matching to the given cell.
        :param semantic_actions: array of semantic actions
        :param cell:
        :return: the action index or None if the action does not exist
        """
        action_ind = [i for i, action in enumerate(semantic_actions) if
                      action.cell[LAT_CELL] == cell[LAT_CELL] and action.cell[LON_CELL] == cell[LON_CELL]]
        if len(action_ind) > 0:
            return action_ind[0]
        else:
            return None

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: SemanticActionsGridState,
                                   action_spec: SemanticActionSpec,
                                   navigation_plan: NavigationPlanMsg) -> TrajectoryParams:
        """
        Generate trajectory specification for trajectory planner given a SemanticActionSpec. This also
        generates the reference route that will be provided to the trajectory planner.
         Given the target longitude and latitude, we create a reference route in global coordinates, where:
         latitude is constant and equal to the target latitude;
         longitude starts from ego current longitude, and end in the target longitude.
        :param behavioral_state: processed behavioral state
        :param navigation_plan: navigation plan of the rest of the roads to be followed (used to create a ref. route)
        :return: Trajectory cost specifications [TrajectoryParameters]
        """
        ego = behavioral_state.ego_state

        # TODO: fix that compensation that once used to be multiplied by the relative_s from ego.
        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = action_spec.s * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        # TODO: figure out how to solve the issue of lagging ego-vehicle (relative to reference route)
        # TODO: better than sending the whole road. and also what happens in the beginning of a road
        center_lane_reference_route = MapService.get_instance().get_uniform_path_lookahead(
            road_id=ego.road_localization.road_id,
            lat_shift=action_spec.d,  # THIS ASSUMES THE GOAL ALWAYS FALLS ON THE REFERENCE ROUTE
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)

        # The frenet frame used in specify (RightHandSide of road)
        rhs_reference_route = MapService.get_instance().get_uniform_path_lookahead(
            road_id=ego.road_localization.road_id,
            lat_shift=0,
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)
        rhs_frenet = FrenetSerret2DFrame(rhs_reference_route)

        # Get road details
        road_id = ego.road_localization.road_id

        # Convert goal state from rhs-frenet-frame to center-lane-frenet-frame
        goal_cstate = rhs_frenet.fstate_to_cstate(np.array([action_spec.s, action_spec.v, 0, action_spec.d, 0, 0]))

        cost_params = SemanticActionsGridPolicy._generate_cost_params(
            road_id=road_id,
            ego_size=ego.size,
            reference_route_latitude=action_spec.d  # this assumes the target falls on the reference route
        )

        trajectory_parameters = TrajectoryParams(reference_route=center_lane_reference_route,
                                                 time=action_spec.t + ego.timestamp_in_sec,
                                                 target_state=goal_cstate,
                                                 cost_params=cost_params,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        return trajectory_parameters

    @staticmethod
    def _generate_cost_params(road_id: int, ego_size: ObjectSize, reference_route_latitude: float) -> \
            TrajectoryCostParams:
        """
        Generate cost specification for trajectory planner
        :param road_id: the road's id - it currently assumes a single road for the whole action.
        :param ego_size: ego size used to extract margins (for dilation of other objects on road)
        :param reference_route_latitude: the latitude of the reference route. This is used to compute out-of-lane cost
        :return: a TrajectoryCostParams instance that encodes all parameters for TP cost computation.
        """
        road = MapService.get_instance().get_road(road_id)
        target_lane_num = int(reference_route_latitude / road.lane_width)

        # lateral distance in [m] from ref. path to rightmost edge of lane
        right_lane_offset = max(0.0, reference_route_latitude - ego_size.width / 2 - target_lane_num * road.lane_width)
        # lateral distance in [m] from ref. path to leftmost edge of lane
        left_lane_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 - \
                           (road.lanes_num - target_lane_num - 1) * road.lane_width
        # as stated above, for shoulders
        right_shoulder_offset = reference_route_latitude - ego_size.width / 2 + SHOULDER_SIGMOID_OFFSET
        # as stated above, for shoulders
        left_shoulder_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 + \
                               SHOULDER_SIGMOID_OFFSET
        # as stated above, for whole road including shoulders
        right_road_offset = reference_route_latitude - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = (road.road_width - reference_route_latitude) - ego_size.width / 2 + ROAD_SHOULDERS_WIDTH

        # Set road-structure-based cost parameters
        right_lane_cost = SigmoidFunctionParams(w=DEVIATION_FROM_LANE_COST, k=LANE_SIGMOID_K_PARAM,
                                                offset=right_lane_offset)  # Zero cost
        left_lane_cost = SigmoidFunctionParams(w=DEVIATION_FROM_LANE_COST, k=LANE_SIGMOID_K_PARAM,
                                               offset=left_lane_offset)  # Zero cost
        right_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=SHOULDER_SIGMOID_K_PARAM,
                                                    offset=right_shoulder_offset)  # Very high cost
        left_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=SHOULDER_SIGMOID_K_PARAM,
                                                   offset=left_shoulder_offset)  # Very high cost
        right_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                                offset=right_road_offset)  # Very high cost
        left_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                               offset=left_road_offset)  # Very high cost

        # Set objects parameters
        # dilate each object by ego length + safety margin
        objects_dilation_length = ego_size.length / 2 + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        objects_dilation_width = ego_size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_cost_x = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=objects_dilation_length)  # Very high (inf) cost
        objects_cost_y = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                               offset=objects_dilation_width)  # Very high (inf) cost
        dist_from_goal_cost = SigmoidFunctionParams(w=DEVIATION_FROM_GOAL_COST, k=GOAL_SIGMOID_K_PARAM,
                                                    offset=GOAL_SIGMOID_OFFSET)
        dist_from_goal_lat_factor = DEVIATION_FROM_GOAL_LAT_LON_RATIO

        cost_params = TrajectoryCostParams(obstacle_cost_x=objects_cost_x,
                                           obstacle_cost_y=objects_cost_y,
                                           left_lane_cost=left_lane_cost,
                                           right_lane_cost=right_lane_cost,
                                           left_shoulder_cost=left_shoulder_cost,
                                           right_shoulder_cost=right_shoulder_cost,
                                           left_road_cost=left_road_cost,
                                           right_road_cost=right_road_cost,
                                           dist_from_goal_cost=dist_from_goal_cost,
                                           dist_from_goal_lat_factor=dist_from_goal_lat_factor,
                                           lon_jerk_cost=LON_JERK_COST,
                                           lat_jerk_cost=LAT_JERK_COST,
                                           velocity_limits=VELOCITY_LIMITS,
                                           lon_acceleration_limits=LON_ACC_LIMITS,
                                           lat_acceleration_limits=LAT_ACC_LIMITS)

        return cost_params

from typing import List, Optional

import numpy as np

from decision_making.src.exceptions import BehavioralPlanningException
from decision_making.src.exceptions import NoValidTrajectoriesFound, raises
from decision_making.src.global_constants import BP_SPECIFICATION_T_MIN, BP_SPECIFICATION_T_MAX, \
    BP_SPECIFICATION_T_RES, A_LON_MIN, \
    A_LON_MAX, A_LAT_MIN, A_LAT_MAX, SAFE_DIST_TIME_DELAY, SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_RIGHT, MIN_OVERTAKE_VEL, \
    BEHAVIORAL_PLANNING_HORIZON, A_LON_EPS, INFINITE_SIGMOID_COST, DEVIATION_FROM_ROAD_COST, DEVIATION_TO_SHOULDER_COST, \
    OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, OBJECTS_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_LON_COST, \
    DEVIATION_FROM_GOAL_LAT_COST, DEVIATION_FROM_REF_ROUTE_COST, LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.global_constants import EGO_ORIGIN_LON_FROM_REAR, TRAJECTORY_ARCLEN_RESOLUTION, \
    PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, VELOCITY_LIMITS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import \
    SemanticActionsGridState
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticAction, SemanticActionSpec, SemanticActionType, \
    LAT_CELL, LON_CELL, SemanticGridCell
from decision_making.src.planning.behavioral.policies.semantic_actions_utils import SemanticActionsUtils as SAU
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import CURVE_X, CURVE_Y
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, ObjectSize
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class SemanticActionsGridPolicy(SemanticActionsPolicy):
    def plan(self, state: State, nav_plan: NavigationPlanMsg):

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = SemanticActionsGridState.create_from_state(state=state,
                                                                      logger=self.logger)

        # iterate over the semantic grid and enumerate all relevant HL actions
        semantic_actions = self._enumerate_actions(behavioral_state=behavioral_state)

        # iterate over all HL actions and generate a specification (desired terminal: position, velocity, time-horizon)
        actions_spec = [self._specify_action(behavioral_state=behavioral_state, semantic_action=semantic_actions[idx],
                                             nav_plan=nav_plan)
                        for idx in range(len(semantic_actions))]

        # Filter actions with invalid spec
        valid_spec_indices = [x for x in range(len(actions_spec)) if actions_spec[x] is not None]
        semantic_actions = [semantic_actions[x] for x in valid_spec_indices]
        actions_spec = [actions_spec[x] for x in valid_spec_indices]

        # evaluate all action-specifications by computing a cost for each action
        action_costs = self._eval_actions(behavioral_state=behavioral_state, semantic_actions=semantic_actions,
                                          actions_spec=actions_spec)

        # select an action-specification with minimal cost
        selected_action_index = int(np.argmin(action_costs))
        selected_action_spec = actions_spec[selected_action_index]

        # translate the selected action-specification into a full specification for the TP
        reference_trajectory = SemanticActionsGridPolicy._generate_reference_route(behavioral_state=behavioral_state,
                                                                                   action_spec=selected_action_spec,
                                                                                   navigation_plan=nav_plan)
        trajectory_parameters = SemanticActionsGridPolicy._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                     action_spec=selected_action_spec,
                                                                                     reference_route=reference_trajectory)

        visualization_message = BehavioralVisualizationMsg(reference_route=reference_trajectory)

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
                    target_obj = behavioral_state.road_occupancy_grid[semantic_cell][0]
                else:
                    # There are no objects in cell
                    target_obj = None

                semantic_action = SemanticAction(cell=semantic_cell, target_obj=target_obj,
                                                 action_type=SemanticActionType.FOLLOW)

                semantic_actions.append(semantic_action)

        return semantic_actions

    def _specify_action(self, behavioral_state: SemanticActionsGridState,
                        semantic_action: SemanticAction, nav_plan: NavigationPlanMsg) -> Optional[SemanticActionSpec]:
        """
        For each semantic action, generate a trajectory specifications that will be passed through to the TP
        :param behavioral_state: semantic actions grid behavioral state
        :param semantic_action:
        :param nav_plan: the navigation plan of ego
        :return: semantic action spec (None if no valid trajectories can be found)
        """

        try:
            if semantic_action.target_obj is None:
                return SemanticActionsGridPolicy._specify_action_to_empty_cell(behavioral_state=behavioral_state,
                                                                               semantic_action=semantic_action)
            else:
                return SemanticActionsGridPolicy._specify_action_towards_object(behavioral_state=behavioral_state,
                                                                                semantic_action=semantic_action,
                                                                                navigation_plan=nav_plan,
                                                                                predictor=self._predictor)
        except NoValidTrajectoriesFound as e:
            return None

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
        is_forward_left_faster = left_lane_action_ind is not None and (current_lane_action_ind is None or
                                                                       actions_spec[left_lane_action_ind].v -
                                                                       actions_spec[
                                                                           current_lane_action_ind].v >= MIN_OVERTAKE_VEL)
        # boolean whether the left cell near ego is occupied
        if (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            is_left_occupied = len(behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_LEFT,
                                                                         SEMANTIC_CELL_LON_SAME)]) > 0
        else:
            is_left_occupied = True

        costs = np.ones(len(semantic_actions))

        # move right if both straight and right lanes are fast
        # if is_forward_right_fast and (is_forward_fast or current_lane_action_ind is None) and not is_right_occupied:
        if is_forward_right_fast and not is_right_occupied:
            costs[right_lane_action_ind] = 0.
        # move left if straight is slow and the left is faster than straight
        elif not is_forward_fast and (
                    is_forward_left_faster or current_lane_action_ind is None) and not is_left_occupied:
            costs[left_lane_action_ind] = 0.
        else:
            costs[current_lane_action_ind] = 0.
        return costs

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: SemanticActionsGridState,
                                   action_spec: SemanticActionSpec, reference_route: np.ndarray) -> TrajectoryParams:
        """
        Generate trajectory specification for trajectory planner given a SemanticActionSpec
        :param behavioral_state: processed behavioral state
        :param reference_route: [nx4] numpy array of (x, y, z, yaw) states
        :return: Trajectory cost specifications [TrajectoryParameters]
        """

        # Get road details
        road_id = behavioral_state.ego_state.road_localization.road_id

        # TODO: should be replaced with cached road statistics on future feature
        frenet = FrenetSerret2DFrame(reference_route[:, [CURVE_X, CURVE_Y]])

        # Create target state
        target_latitude = behavioral_state.ego_state.road_localization.intra_road_lat + action_spec.d_rel
        target_longitude = behavioral_state.ego_state.road_localization.road_lon + action_spec.s_rel

        curvature = float(frenet.get_curvature(target_longitude))

        target_state = frenet.fstate_to_cstate(
            np.array([target_longitude, action_spec.v, 0, target_latitude, 0, curvature]))

        cost_params = SemanticActionsGridPolicy._generate_cost_params(
            road_id=road_id,
            ego_size=behavioral_state.ego_state.size,
            reference_route_latitude=target_latitude  # this assumes the target falls on the reference route
        )

        trajectory_parameters = TrajectoryParams(reference_route=reference_route,
                                                 time=action_spec.t + behavioral_state.ego_state.timestamp_in_sec,
                                                 target_state=target_state,
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
        left_margin = right_margin = ego_size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        # lateral distance in [m] from ref. path to rightmost edge of lane
        right_lane_offset = max(0.0, reference_route_latitude - right_margin - target_lane_num * road.lane_width)
        # lateral distance in [m] from ref. path to leftmost edge of lane
        left_lane_offset = (road.road_width - reference_route_latitude) - left_margin - \
                           (road.lanes_num - target_lane_num - 1) * road.lane_width
        # as stated above, for shoulders
        right_shoulder_offset = reference_route_latitude - right_margin
        # as stated above, for shoulders
        left_shoulder_offset = (road.road_width - reference_route_latitude) - left_margin
        # as stated above, for whole road including shoulders
        right_road_offset = right_shoulder_offset + ROAD_SHOULDERS_WIDTH
        # as stated above, for whole road including shoulders
        left_road_offset = left_shoulder_offset + ROAD_SHOULDERS_WIDTH

        # Set road-structure-based cost parameters
        right_lane_cost = SigmoidFunctionParams(w=OUT_OF_LANE_COST, k=ROAD_SIGMOID_K_PARAM,
                                                offset=right_lane_offset)  # Zero cost
        left_lane_cost = SigmoidFunctionParams(w=OUT_OF_LANE_COST, k=ROAD_SIGMOID_K_PARAM,
                                               offset=left_lane_offset)  # Zero cost
        right_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=ROAD_SIGMOID_K_PARAM,
                                                    offset=right_shoulder_offset)  # Very high cost
        left_shoulder_cost = SigmoidFunctionParams(w=DEVIATION_TO_SHOULDER_COST, k=ROAD_SIGMOID_K_PARAM,
                                                   offset=left_shoulder_offset)  # Very high cost
        right_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                                offset=right_road_offset)  # Very high cost
        left_road_cost = SigmoidFunctionParams(w=DEVIATION_FROM_ROAD_COST, k=ROAD_SIGMOID_K_PARAM,
                                               offset=left_road_offset)  # Very high cost

        # Set objects parameters
        # dilate each object by ego length + safety margin
        objects_dilation_size = ego_size.length + LATERAL_SAFETY_MARGIN_FROM_OBJECT
        objects_cost = SigmoidFunctionParams(w=INFINITE_SIGMOID_COST, k=OBJECTS_SIGMOID_K_PARAM,
                                             offset=objects_dilation_size)  # Very high (inf) cost

        dist_from_goal_lon_sq_cost = DEVIATION_FROM_GOAL_LON_COST
        dist_from_goal_lat_sq_cost = DEVIATION_FROM_GOAL_LAT_COST
        dist_from_ref_sq_cost = DEVIATION_FROM_REF_ROUTE_COST

        # TODO: set velocity and acceleration limits properly
        velocity_limits = VELOCITY_LIMITS  # [m/s]. not a constant because it might be learned. TBD
        acceleration_limits = np.array([A_LON_MIN - A_LON_EPS, A_LON_MAX + A_LON_EPS])

        cost_params = TrajectoryCostParams(left_lane_cost=left_lane_cost,
                                           right_lane_cost=right_lane_cost,
                                           left_road_cost=left_road_cost,
                                           right_road_cost=right_road_cost,
                                           left_shoulder_cost=left_shoulder_cost,
                                           right_shoulder_cost=right_shoulder_cost,
                                           obstacle_cost=objects_cost,
                                           dist_from_goal_lon_sq_cost=dist_from_goal_lon_sq_cost,
                                           dist_from_goal_lat_sq_cost=dist_from_goal_lat_sq_cost,
                                           dist_from_ref_sq_cost=dist_from_ref_sq_cost,
                                           velocity_limits=velocity_limits,
                                           acceleration_limits=acceleration_limits)

        return cost_params

    # TODO: rethink the design of this function
    @staticmethod
    def _specify_action_to_empty_cell(behavioral_state: SemanticActionsGridState,
                                      semantic_action: SemanticAction) -> SemanticActionSpec:
        """
        This method's purpose is to specify the enumerated actions that the agent can take.
        Each semantic action is translated to a trajectory of the agent.
        The trajectory specification is created towards a target location/object in given cell,
         considering ego speed, location.
        :param behavioral_state:
        :param semantic_action:
        :return: semantic action specification
        """
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(
            road_id=behavioral_state.ego_state.road_localization.road_id)
        target_lane = behavioral_state.ego_state.road_localization.lane_num + semantic_action.cell[LAT_CELL]
        target_lane_latitude = road_lane_latitudes[target_lane]

        # We set the desired longitudinal distance to be equal to the distance we
        # would have travelled for the planning horizon in the average speed between current and target vel.
        target_relative_s = SAU.compute_distance_by_velocity_diff(behavioral_state.ego_state.v_x)
        target_relative_d = target_lane_latitude - behavioral_state.ego_state.road_localization.intra_road_lat

        return SemanticActionSpec(t=BEHAVIORAL_PLANNING_HORIZON, v=BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED,
                                  s_rel=target_relative_s, d_rel=target_relative_d)

    @staticmethod
    @raises(NoValidTrajectoriesFound)
    # TODO: modify this function to work with DynamicObject's specific NavigationPlan (and predictor?)
    def _specify_action_towards_object(behavioral_state: SemanticActionsGridState,
                                       semantic_action: SemanticAction,
                                       navigation_plan: NavigationPlanMsg,
                                       predictor: Predictor) -> SemanticActionSpec:
        """
        given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec
        :param behavioral_state: semantic actions grid behavioral state
        :param semantic_action:
        :return: SemanticActionSpec
        """
        # Extract relevant details from state on Ego
        ego_v_x = behavioral_state.ego_state.v_x
        ego_v_y = behavioral_state.ego_state.v_y

        ego_on_road = behavioral_state.ego_state.road_localization
        ego_theta_diff = ego_on_road.intra_road_yaw  # relative to road

        ego_sx0 = 0.0
        ego_sv0 = np.cos(ego_theta_diff) * ego_v_x + np.sin(ego_theta_diff) * ego_v_y
        ego_sa0 = 0.0  # TODO: to be changed to include acc

        ego_dx0 = ego_on_road.intra_road_lat
        ego_dv0 = -np.sin(ego_theta_diff) * ego_v_x + np.cos(ego_theta_diff) * ego_v_y
        ego_da0 = 0.0  # TODO: to be changed to include acc

        # Extract relevant details from state on Reference-Object
        obj_on_road = semantic_action.target_obj.road_localization
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]
        # TODO: rotate speed v_x, v_y to road coordinated to get the actual lon/lat speed
        # obj_v_x = semantic_action.target_obj.road_longitudinal_speed
        # obj_v_y = semantic_action.target_obj.road_lateral_speed
        # obj_theta_diff = obj_on_road.intra_road_yaw  # relative to road

        # TODO: use navigation plan of the object
        # object sx0 is relative to ego
        obj_sx0 = MapService.get_instance().get_longitudinal_difference(ego_on_road.road_id, ego_on_road.road_lon,
                                                      obj_on_road.road_id, obj_on_road.road_lon, navigation_plan)
        obj_sv0 = semantic_action.target_obj.road_longitudinal_speed
        obj_sa0 = 0.0  # TODO: to be changed to include acc

        # lon_margin = part of ego from its origin to its front + half of target object
        lon_margin = behavioral_state.ego_state.size.length - EGO_ORIGIN_LON_FROM_REAR + \
                     semantic_action.target_obj.size.length/2

        prediction_timestamps = np.arange(BP_SPECIFICATION_T_MIN, BP_SPECIFICATION_T_MAX, BP_SPECIFICATION_T_RES)

        for T in prediction_timestamps:
            # TODO: should be cached in advance using OCU.QP1D.time_constraints_tensor
            A = OptimalControlUtils.QuinticPoly1D.time_constraints_matrix(T)
            A_inv = np.linalg.inv(A)

            # TODO: should be swapped with current implementation of Predictor
            obj_saT = obj_sa0
            obj_svT = obj_sv0 + obj_sa0 * T
            obj_sxT = obj_sx0 + obj_sv0 * T + obj_sa0 * T ** 2 / 2

            # TODO: account for acc<>0 (from MobilEye's paper)
            safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY

            # set of 6 constraints RHS values for quintic polynomial solution (S DIM)
            constraints_s = np.array(
                [ego_sx0, ego_sv0, ego_sa0, obj_sxT - safe_lon_dist - lon_margin, obj_svT, obj_saT])
            constraints_d = np.array([ego_dx0, ego_dv0, ego_da0, obj_center_lane_latitude, 0.0, 0.0])

            # solve for s(t) and d(t)
            poly_coefs_s = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
            poly_coefs_d = OptimalControlUtils.QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

            # TODO: acceleration is computed in frenet frame and not cartesian. if road is curved, this is problematic
            if SemanticActionsGridPolicy._is_acceleration_in_limits(poly_coefs_s, T, A_LON_MIN, A_LON_MAX) and \
                    SemanticActionsGridPolicy._is_acceleration_in_limits(poly_coefs_d, T, A_LAT_MIN, A_LAT_MAX):
                return SemanticActionSpec(t=T, v=obj_svT, s_rel=constraints_s[3] - ego_sx0,
                                          d_rel=constraints_d[3] - ego_dx0)

        raise NoValidTrajectoriesFound("No valid trajectories found. action: %s, state: %s, ",
                                       semantic_action.__dict__, behavioral_state.__dict__)

    @staticmethod
    def _is_acceleration_in_limits(poly_coefs: np.ndarray, T: float,
                                   min_acc_threshold: float, max_acc_threshold: float) -> bool:
        """
        given a quintic polynomial coefficients vector, and restrictions
        on the acceleration values, return True if restrictions are met, False otherwise
        :param poly_coefs: 1D numpy array with s(t), s_dot(t) s_dotdot(t) concatenated
        :param T: planning time horizon [sec]
        :param min_acc_threshold: minimal allowed value of acceleration/deceleration [m/sec^2]
        :param max_acc_threshold: maximal allowed value of acceleration/deceleration [m/sec^2]
        :return: True if restrictions are met, False otherwise
        """
        # TODO: a(0) and a(T) checks are omitted as they they are provided by the user.
        # compute extrema points, by finding the roots of the 3rd derivative (which is itself a 2nd degree polynomial)
        acc_suspected_points_s = np.roots(np.polyder(poly_coefs, m=3))
        acceleration_poly_coefs = np.polyder(poly_coefs, m=2)
        acc_suspected_values_s = np.polyval(acceleration_poly_coefs, acc_suspected_points_s)

        # filter out extrema points out of [0, T]
        acc_inlimit_suspected_values_s = acc_suspected_values_s[np.greater_equal(acc_suspected_points_s, 0) &
                                                                np.less_equal(acc_suspected_points_s, T)]

        # check if extrema values are within [a_min, a_max] limits
        return np.all(np.greater_equal(acc_inlimit_suspected_values_s, min_acc_threshold) &
                      np.less_equal(acc_inlimit_suspected_values_s, max_acc_threshold))

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
    def _generate_reference_route(behavioral_state: SemanticActionsGridState,
                                  action_spec: SemanticActionSpec, navigation_plan: NavigationPlanMsg) -> np.ndarray:
        """
        Generate the reference route that will be provided to the trajectory planner.
         Given the target longitude and latitude, we create a reference route in global coordinates, where:
         latitude is constant and equal to the target latitude;
         longitude starts from ego current longitude, and end in the target longitude.
        :param behavioral_state: processed behavioral state
        :param action_spec: the goal of the action
        :return: [nx3] array of reference_route (x,y,yaw) [m,m,rad] in global coordinates
        """

        target_lane_latitude = action_spec.d_rel + behavioral_state.ego_state.road_localization.intra_road_lat
        target_relative_longitude = action_spec.s_rel

        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = behavioral_state.ego_state.road_localization.road_lon + \
                             target_relative_longitude * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        # TODO: figure out how to solve the issue of lagging ego-vehicle (relative to reference route)
        # TODO: better than sending the whole road. and also what happens in the begginning of a road
        lookahead_path = MapService.get_instance().get_uniform_path_lookahead(
            road_id=behavioral_state.ego_state.road_localization.road_id,
            lat_shift=target_lane_latitude,
            starting_lon=0,
            lon_step=TRAJECTORY_ARCLEN_RESOLUTION,
            steps_num=int(np.ceil(lookahead_distance / TRAJECTORY_ARCLEN_RESOLUTION)),
            navigation_plan=navigation_plan)

        return lookahead_path



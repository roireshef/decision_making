import numpy as np

from decision_making.src.planning.behavioral.architecture.components.state_action_evaluator import StateActionEvaluator
from decision_making.src.planning.behavioral.architecture.components.value_approximator import ValueApproximator

from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, TRAJECTORY_ARCLEN_RESOLUTION, \
    SHOULDER_SIGMOID_OFFSET, DEVIATION_FROM_LANE_COST, LANE_SIGMOID_K_PARAM, SHOULDER_SIGMOID_K_PARAM, \
    DEVIATION_TO_SHOULDER_COST, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, OBSTACLE_SIGMOID_COST, \
    OBSTACLE_SIGMOID_K_PARAM, DEVIATION_FROM_GOAL_COST, GOAL_SIGMOID_K_PARAM, GOAL_SIGMOID_OFFSET, \
    DEVIATION_FROM_GOAL_LAT_LON_RATIO, LON_JERK_COST, LAT_JERK_COST, VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams, TrajectoryCostParams, \
    SigmoidFunctionParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.architecture.components.action_space import ActionSpace
from decision_making.src.planning.behavioral.architecture.components.action_validator import ActionValidator
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import SemanticActionsGridState
from decision_making.src.planning.behavioral.policies.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, ObjectSize, EgoState
from logging import Logger
from typing import Optional

from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from mapping.src.service.map_service import MapService


class BehavioralPlanner:
    def __init__(self, action_space: ActionSpace, state_action_evaluator: StateActionEvaluator,
                 action_validator: ActionValidator, value_approximator: ValueApproximator,
                 predictor: Predictor, logger: Logger):
        self.action_space = action_space
        self.state_action_evaluator = state_action_evaluator
        self.action_validator = action_validator
        self.value_approximator = value_approximator
        self.predictor = predictor
        self.logger = logger

        self._last_ego_state: Optional[EgoState] = None
        self._last_action: Optional[ActionRecipe] = None
        self._last_action_spec: Optional[ActionSpec] = None
        self._last_poly_coefs_s: Optional[np.ndarray] = None

    def plan(self, state: State, nav_plan: NavigationPlanMsg):
        """
        Given current state and navigation plan, plans the next semantic action to be carried away. This method makes
        use of Planner components such as Evaluator,Validator and Predictor for enumerating, specifying
        and evaluating actions. Its output will be further handled and used to create a trajectory in Trajectory Planner
        and has the form of TrajectoryParams, which includes the reference route, target time, target state to be in,
        cost params and strategy.
        :param state:
        :param nav_plan:
        :return: a tuple: (TrajectoryParams for TP,visualization_message for e.g. VizTool)
        """
        pass


class CostBasedBehavioralPlanner(BehavioralPlanner):
    def __init__(self, action_space: ActionSpace, action_evaluator: StateActionEvaluator, action_validator: ActionValidator,
                 value_approximator: ValueApproximator, predictor: Predictor, logger: Logger):
        super().__init__(action_space, action_evaluator, action_validator, value_approximator, predictor, logger)

    def plan(self, state: State, nav_plan: NavigationPlanMsg):

        action_recipes = self.action_space.recipes

        # create road semantic grid from the raw State object
        # behavioral_state contains road_occupancy_grid and ego_state
        behavioral_state = SemanticActionsGridState.create_from_state(state=state,
                                                                      logger=self.logger)

        current_state_value = self.value_approximator.evaluate_state(behavioral_state)

        # recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        recipes_cost = self.state_action_evaluator.evaluate_recipes(behavioral_state, action_recipes, recipes_mask)

        action_specs = []

        for i in range(len(action_recipes)):
            spec = None
            if recipes_mask[i]:
                consistent_behavioral_state = CostBasedBehavioralPlanner._maintain_consistency(self, behavioral_state, action_recipes[i])
                spec = self.action_space.specify_goal(action_recipes[i], consistent_behavioral_state)

            action_specs.append(spec)

        action_specs_mask = self.action_validator.validate_actions(action_specs, behavioral_state)

        action_costs = self.state_action_evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask)

        selected_action_index = int(np.argmin(action_costs))
        selected_action_spec = action_specs[selected_action_index]

        trajectory_parameters = CostBasedBehavioralPlanner._generate_trajectory_specs(behavioral_state=behavioral_state,
                                                                                      action_spec=selected_action_spec,
                                                                                      navigation_plan=nav_plan)
        visualization_message = BehavioralVisualizationMsg(reference_route=trajectory_parameters.reference_route)

        # updating selected actions in memory
        self._last_action = action_recipes[selected_action_index]
        self._last_action_spec = selected_action_spec
        self._last_ego_state = state.ego_state

        self.logger.debug("Chosen behavioral semantic action is %s, %s",
                          action_recipes[selected_action_index].__dict__, selected_action_spec.__dict__)

        return trajectory_parameters, visualization_message

    @staticmethod
    def _generate_trajectory_specs(behavioral_state: SemanticActionsGridState,
                                   action_spec: ActionSpec,
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

        # Add a margin to the lookahead path of dynamic objects to avoid extrapolation
        # caused by the curve linearization approximation in the resampling process
        # The compensation here is multiplicative because of the different curve-fittings we use:
        # in BP we use piecewise-linear and in TP we use cubic-fit.
        # Due to that, a point's longitude-value will be different between the 2 curves.
        # This error is accumulated depending on the actual length of the curvature -
        # when it is long, the error will potentially be big.
        lookahead_distance = action_spec.s * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        # TODO: figure out how to solve the issue of lagging ego-vehicle (relative to reference route)
        # TODO: better than sending the whole road. Fix when map service is redesigned!
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

        cost_params = CostBasedBehavioralPlanner._generate_cost_params(
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
        objects_dilation_length = SemanticActionsUtils.get_ego_lon_margin(ego_size)
        objects_dilation_width = SemanticActionsUtils.get_ego_lat_margin(ego_size)
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

    def _maintain_consistency(self, behavioral_state: SemanticActionsGridState,
                              action_recipe: ActionRecipe) -> SemanticActionsGridState:
        """
        Updates behavioral_state in order to maintain consistency.
        :param behavioral_state: processed behavioral state
        :param action_recipe:
        :return: updated behavioral_state [SemanticActionsGridState]
        """
        ego_state = behavioral_state.ego_state

        consistent_behavioral_state = behavioral_state

        # BP IF - if ego is close to last planned trajectory (in BP), then assume ego is exactly on this trajectory
        if self._last_action is not None and action_recipe == self._last_action \
                and LocalizationUtils.is_actual_state_close_to_expected_state(
            ego_state, self._last_action_spec.samplable_trajectory, self.logger, self.__class__.__name__):

            ego_cstate = self._last_action_spec.samplable_trajectory.sample(np.array([ego_state.timestamp_in_sec]))[0]

            new_ego_state = ego_state
            new_ego_state.x = ego_cstate[0]
            new_ego_state.y = ego_cstate[1]
            new_ego_state.yaw = ego_cstate[2]
            new_ego_state.v_x = ego_cstate[3]
            new_ego_state.acceleration_lon = ego_cstate[4]
            new_ego_state.steering_angle = np.arctan(new_ego_state.size.length * ego_cstate[5])  # steering_angle=atan(ego_len*curvature)
            new_ego_state.omega_yaw = new_ego_state.curvature * new_ego_state.v_x

            consistent_behavioral_state.ego_state = new_ego_state

        return consistent_behavioral_state

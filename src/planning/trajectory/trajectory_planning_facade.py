import time

import numpy as np
import traceback
from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from decision_making.src.exceptions import MsgDeserializationError, NoValidTrajectoriesFound, StateHasNotArrivedYet
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, \
    TRAJECTORY_PLANNING_NAME_FOR_METRICS, MAX_TRAJECTORY_WAYPOINTS, TRAJECTORY_WAYPOINT_SIZE, \
    LOG_MSG_SCENE_STATIC_RECEIVED, VISUALIZATION_PREDICTION_RESOLUTION, MAX_NUM_POINTS_FOR_VIZ, \
    MAX_VIS_TRAJECTORIES_NUMBER, LOG_MSG_TRAJECTORY_PLAN_FROM_DESIRED, LOG_MSG_TRAJECTORY_PLAN_FROM_ACTUAL, \
    BP_ACTION_T_LIMITS
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlan, DataTrajectoryPlan
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg, \
    PredictionsVisualization, DataTrajectoryVisualization
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import CartesianTrajectories, FP_SX, C_Y, FS_DX, \
    FS_SX, FrenetState2D, C_X, FS_SV, FS_SA, FS_DV, FS_DA, CartesianExtendedState
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, EgoState
from decision_making.src.utils.metric_logger import MetricLogger
from logging import Logger
from typing import Dict
import copy


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 last_trajectory: SamplableTrajectory = None):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers - types are
        {TrajectoryPlanningStrategy: TrajectoryPlanner}
        :param last_trajectory: a representation of the last trajectory that was planned during self._periodic_action_impl
        """
        super().__init__(pubsub=pubsub, logger=logger)

        MetricLogger.init(TRAJECTORY_PLANNING_NAME_FOR_METRICS)

        self._strategy_handlers = strategy_handlers
        self._validate_strategy_handlers()
        self._last_trajectory = last_trajectory
        self._started_receiving_states = False

    def _start_impl(self):
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_PARAMS_LCM"], None)
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"], None)
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"], None)

    def _stop_impl(self):
        self.pubsub.unsubscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_PARAMS_LCM"])
        self.pubsub.unsubscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"])
        self.pubsub.unsubscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"])

    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            # Monitor execution time of a time-critical component (prints to logging at the end of method)
            start_time = time.time()

            scene_static = self._get_current_scene_static()
            SceneStaticModel.get_instance().set_scene_static(scene_static)

            state = self._get_current_state()

            params = self._get_mission_params()

            # Longitudinal planning horizon (Ts)
            lon_plan_horizon = params.time - state.ego_state.timestamp_in_sec
            minimal_required_horizon = float(params.bp_time)/1e9 + BP_ACTION_T_LIMITS[0] - state.ego_state.timestamp_in_sec

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route.points[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: velocity: %s", state.ego_state.velocity)
            self.logger.debug("TrajectoryPlanningFacade is required to plan with time horizon = %s", lon_plan_horizon)
            self.logger.debug("state: %d objects detected", len(state.dynamic_objects))

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                sampled_state = self._get_state_with_expected_ego(state) if self._last_trajectory is not None else None
                # TODO: remove it
                ego_fstate = state.ego_state.map_state.lane_fstate
                sampled_cartesian = sampled_state.ego_state.cartesian_state
                sampled_fstate = sampled_state.ego_state.map_state.lane_fstate
                ego_time = state.ego_state.timestamp_in_sec
                dist_to_goal = np.linalg.norm(params.target_state[:2] - sampled_cartesian[:2])
                time_to_goal = params.time - ego_time
                print('TP if: time %.3f; orig-fstate: (%.2f, %.3f, %.2f) -> (%.2f, %.3f, %.2f,  %.3f, %.3f, %.2f); '
                      'cpoint: (%.2f, %.2f); to_goal: t=%.3f s=%.3f s/t=%.3f' %
                      (ego_time, ego_fstate[FS_SX], ego_fstate[FS_SV], ego_fstate[FS_SA],
                       sampled_fstate[FS_SX], sampled_fstate[FS_SV], sampled_fstate[FS_SA],
                       sampled_fstate[FS_DX], sampled_fstate[FS_DV], sampled_fstate[FS_DA],
                       sampled_cartesian[C_X], sampled_cartesian[C_Y], time_to_goal, dist_to_goal, dist_to_goal/time_to_goal))
                self.logger.debug(LOG_MSG_TRAJECTORY_PLAN_FROM_DESIRED,
                                  sampled_state.ego_state.map_state,
                                  state.ego_state.map_state)
                updated_state = sampled_state
            else:
                # TODO: remove it
                ego_fstate = state.ego_state.map_state.lane_fstate
                print('TP else: time %.3f; orig-fstate: (%.2f, %.3f, %.2f); T_s=%f' %
                      (state.ego_state.timestamp_in_sec, ego_fstate[FS_SX], ego_fstate[FS_SV], ego_fstate[FS_SA],
                       params.time - state.ego_state.timestamp_in_sec))
                self.logger.warning(LOG_MSG_TRAJECTORY_PLAN_FROM_ACTUAL, state.ego_state.map_state)
                updated_state = state

            MetricLogger.get_logger().bind(bp_time=params.bp_time)

            # project all dynamic objects on the reference route
            projected_obj_fstates = TrajectoryPlanningFacade._project_objects_on_reference_route(
                updated_state, params.reference_route)

            # plan a trajectory according to specification from upper DM level
            samplable_trajectory, ctrajectories, _ = self._strategy_handlers[params.strategy]. \
                plan(updated_state, params.reference_route, projected_obj_fstates, params.target_state,
                     lon_plan_horizon, minimal_required_horizon, params.cost_params)

            trajectory_msg = self.generate_trajectory_plan(timestamp=state.ego_state.timestamp_in_sec,
                                                           samplable_trajectory=samplable_trajectory)

            self._publish_trajectory(trajectory_msg)
            self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, trajectory_msg)

            # TODO: handle viz for fixed trajectories
            # publish visualization/debug data - based on short term prediction aligned state!
            debug_results = TrajectoryPlanningFacade._prepare_visualization_msg(
                state, projected_obj_fstates, ctrajectories, max(lon_plan_horizon, minimal_required_horizon),
                self._strategy_handlers[params.strategy].predictor, params.reference_route)

            self._publish_debug(debug_results)

            print('---- TP: %f sec' % (time.time() - start_time))

            self.logger.info("%s %s", LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, time.time() - start_time)
            MetricLogger.get_logger().report()

        except StateHasNotArrivedYet:
            self.logger.warning("StateHasNotArrivedYet was raised. skipping planning.")

        except MsgDeserializationError:
            self.logger.warning("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s ",
                              traceback.format_exc())

        # TODO - we need to handle this as an emergency.
        except NoValidTrajectoriesFound:
            self.logger.error("TrajectoryPlanningFacade: NoValidTrajectoriesFound was raised. skipping planning. %s",
                              traceback.format_exc())

        except Exception:
            self.logger.critical("TrajectoryPlanningFacade: UNHANDLED EXCEPTION in trajectory planning: %s",
                                 traceback.format_exc())

    @staticmethod
    def _project_objects_on_reference_route(state: State, reference_route: GeneralizedFrenetSerretFrame) -> \
            Dict[int, FrenetState2D]:
        """
        Project all dynamic objects on the reference route. In case of failure set map_state = None.
        :param state: original state
        :param reference_route: GFF to project on
        :return: dictionary from obj_id to its projected map_states on the reference route
        """
        projected_obj_fstates = {}
        projected_state = copy.deepcopy(state)
        for obj in projected_state.dynamic_objects:
            try:
                projected_obj_fstates[obj.obj_id] = reference_route.cstate_to_fstate(obj.cartesian_state)
            except Exception:  # too far object
                pass
        return projected_obj_fstates

    # TODO: add map_origin that is sent from the outside
    def generate_trajectory_plan(self, timestamp: float, samplable_trajectory: SamplableTrajectory):
        """
        sample trajectory points from the samplable-trajectory, translate them according to ego's reference point and
        wrap them in a message to the controller
        :param timestamp: the timestamp to use as a reference for the beginning of trajectory
        :param samplable_trajectory: the trajectory plan to sample points from (samplable object)
        :return: a TrajectoryPlan message ready to send to the controller
        """
        trajectory_points = samplable_trajectory.sample(
            np.linspace(start=0,
                        stop=(TRAJECTORY_NUM_POINTS - 1) * TRAJECTORY_TIME_RESOLUTION,
                        num=TRAJECTORY_NUM_POINTS) + timestamp)
        self._last_trajectory = samplable_trajectory

        # publish results to the lower DM level (Control)
        # TODO: put real values in tolerance and maximal velocity fields
        # TODO: understand if padding with zeros is necessary
        waypoints = np.vstack((np.hstack((trajectory_points, np.zeros(shape=[TRAJECTORY_NUM_POINTS,
                                                                             TRAJECTORY_WAYPOINT_SIZE -
                                                                             trajectory_points.shape[1]]))),
                               np.zeros(shape=[MAX_TRAJECTORY_WAYPOINTS - TRAJECTORY_NUM_POINTS,
                                               TRAJECTORY_WAYPOINT_SIZE])))

        timestamp_object = Timestamp.from_seconds(timestamp)
        map_origin = MapOrigin(e_phi_latitude=0, e_phi_longitude=0, e_l_altitude=0, s_Timestamp=timestamp_object)

        trajectory_msg = TrajectoryPlan(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,
                                                        e_Cnt_version=0),
                                        s_Data=DataTrajectoryPlan(s_Timestamp=timestamp_object, s_MapOrigin=map_origin,
                                                                  a_TrajectoryWaypoints=waypoints,
                                                                  e_Cnt_NumValidTrajectoryWaypoints=TRAJECTORY_NUM_POINTS))

        return trajectory_msg

    def _validate_strategy_handlers(self) -> None:
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not self._strategy_handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(self._strategy_handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    def _get_current_state(self) -> State:
        """
        Returns the last received world state.
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized State
        """
        is_success, serialized_state = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"], timeout=1)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_state is None:
            if self._started_receiving_states:
                # PubSub queue is empty after being non-empty for a while
                raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t "
                                              "subscribed" % pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"])
            else:
                # Pubsub queue is empty since planning module is up
                raise StateHasNotArrivedYet("Waiting for data from SceneProvider/StateModule")
        self._started_receiving_states = True
        state = State.deserialize(serialized_state)
        self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, state))
        return state

    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"], timeout=1)
        # TODO Move the raising of the exception to pubsub code. Do the same in behavioral facade
        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"])
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        if scene_static.s_Data.e_Cnt_num_lane_segments == 0 and scene_static.s_Data.e_Cnt_num_road_segments == 0:
            raise MsgDeserializationError("SceneStatic map was received without any road or lanes")
        self.logger.debug("%s: %f" % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        is_success, serialized_params = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_PARAMS_LCM"], timeout=1)
        if serialized_params is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed' %
                                          pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_PARAMS_LCM"])
        trajectory_params = TrajectoryParams.deserialize(serialized_params)
        self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, trajectory_params)
        return trajectory_params

    def _publish_trajectory(self, results: TrajectoryPlan) -> None:
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_PLAN"], results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TRAJECTORY_VISUALIZATION"], debug_msg.serialize())

    def _get_state_with_expected_ego(self, state: State) -> State:
        """
        takes a state and overrides its ego vehicle's localization to be the localization expected at the state's
        timestamp according to the last trajectory cached in the facade's self._last_trajectory.
        Note: 1. Lateral velocity is zeroed since we don't plan for drifts and lateral components are being reflected
                 in yaw and curvature.
              2. We assume that self._last_trajectory is WerlingSamplableTrajectory.
        :param state: the state to process
        :return: a new state object with a new ego-vehicle localization
        """
        ego = state.ego_state
        samplable = self._last_trajectory
        expected_fstate = samplable.sample_frenet(np.array([ego.timestamp_in_sec]))[0]
        expected_cstate = samplable.frenet_frame.fstate_to_cstate(expected_fstate)
        lane_id, lane_fstate = samplable.frenet_frame.convert_to_segment_state(expected_fstate)
        map_state = MapState(lane_fstate, lane_id)
        expected_ego_state = EgoState(ego.obj_id, ego.timestamp, expected_cstate, map_state, map_state, ego.size, ego.confidence)
        return state.clone_with(ego_state=expected_ego_state)

    @staticmethod
    def _prepare_visualization_msg(state: State, objects_fstates: Dict[int, FrenetState2D],
                                   ctrajectories: CartesianTrajectories, planning_horizon: float,
                                   predictor: EgoAwarePredictor, reference_route: GeneralizedFrenetSerretFrame) -> \
            TrajectoryVisualizationMsg:
        """
        prepares visualization message for visualization purposes
        :param state: short-term prediction aligned state
        :param ctrajectories: alternative trajectories in cartesian-frame
        :param planning_horizon: [sec] the (relative) planning-horizon used for planning
        :param predictor: predictor for the actors' predictions
        :return: trajectory visualization message
        """
        # TODO: add recipe to trajectory_params for goal's description
        # slice alternative trajectories by skipping indices - for visualization
        alternative_ids_skip_range = np.round(np.linspace(0, len(ctrajectories)-1, MAX_VIS_TRAJECTORIES_NUMBER)).astype(int)
        # slice alternative trajectories by skipping indices - for visualization
        sliced_ctrajectories = ctrajectories[alternative_ids_skip_range]

        # this assumes the state is already aligned by short time prediction
        prediction_horizons = np.arange(0, planning_horizon, VISUALIZATION_PREDICTION_RESOLUTION, float)

        # visualize objects' predictions
        # TODO: create 3 GFFs in TP and convert objects' predictions on them
        objects_visualizations = []
        for obj_id, obj_fstate in objects_fstates.items():
            obj_fpredictions = predictor.predict_2d_frenet_states(np.array([obj_fstate]),
                                                                  prediction_horizons)[0][:, [FS_SX, FS_DX]]
            # skip objects having predictions out of reference_route
            valid_obj_fpredictions = obj_fpredictions[obj_fpredictions[:, FP_SX] < reference_route.s_max]
            obj_cpredictions = reference_route.fpoints_to_cpoints(valid_obj_fpredictions)
            objects_visualizations.append(PredictionsVisualization(obj_id, obj_cpredictions))

        header = Header(0, Timestamp.from_seconds(state.ego_state.timestamp_in_sec), 0)
        trajectory_length = ctrajectories.shape[1]
        points_step = int(trajectory_length / MAX_NUM_POINTS_FOR_VIZ) + 1
        visualization_data = DataTrajectoryVisualization(
            sliced_ctrajectories[:, :trajectory_length:points_step, :(C_Y+1)],  # at most MAX_NUM_POINTS_FOR_VIZ points
            objects_visualizations, "")
        return TrajectoryVisualizationMsg(header, visualization_data)

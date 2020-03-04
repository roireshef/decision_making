import time

import numpy as np
import traceback

from decision_making.src.messages.control_status_message import ControlStatus
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PLAN
from interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS
from interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_DYNAMIC
from interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_VISUALIZATION
from interface.Rte_Types.python.uc_system import UC_SYSTEM_CONTROL_STATUS

from decision_making.src.exceptions import MsgDeserializationError, CartesianLimitsViolated, StateHasNotArrivedYet
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, TRAJECTORY_NUM_POINTS, \
    LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, \
    TRAJECTORY_PLANNING_NAME_FOR_METRICS, MAX_TRAJECTORY_WAYPOINTS, TRAJECTORY_WAYPOINT_SIZE, \
    VISUALIZATION_PREDICTION_RESOLUTION, MAX_NUM_POINTS_FOR_VIZ, \
    MAX_VIS_TRAJECTORIES_NUMBER, REPLANNING_LAT, REPLANNING_LON, \
    LOG_MSG_SCENE_DYNAMIC_RECEIVED, LOG_MSG_CONTROL_STATUS
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlan, DataTrajectoryPlan
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg, \
    PredictionsVisualization, DataTrajectoryVisualization
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import CartesianTrajectories, FP_SX, C_Y, FS_DX,  FS_SX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State
from decision_making.src.utils.dm_profiler import DMProfiler
from decision_making.src.utils.metric_logger.metric_logger import MetricLogger
from logging import Logger
from typing import Dict, Optional
import rte.python.profiler as prof


class TrajectoryPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger,
                 strategy_handlers: Dict[TrajectoryPlanningStrategy, TrajectoryPlanner],
                 last_trajectory: SamplableWerlingTrajectory = None):
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
        self.pubsub.subscribe(UC_SYSTEM_TRAJECTORY_PARAMS)
        self.pubsub.subscribe(UC_SYSTEM_SCENE_DYNAMIC)
        self.pubsub.subscribe(UC_SYSTEM_CONTROL_STATUS)

    def _stop_impl(self):
        self.pubsub.unsubscribe(UC_SYSTEM_TRAJECTORY_PARAMS)
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_DYNAMIC)
        self.pubsub.unsubscribe(UC_SYSTEM_CONTROL_STATUS)

    def _periodic_action_impl(self):
        """
        will execute planning using the implementation for the desired planning-strategy provided
        :return: no return value. results are published in self.__publish_results()
        """
        try:
            # Monitor execution time of a time-critical component (prints to logging at the end of method)
            start_time = time.time()

            params = self._get_mission_params()

            with DMProfiler(self.__class__.__name__ + '._get_current_scene_dynamic'):
                scene_dynamic = self._get_current_scene_dynamic()

                state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic,
                                                              selected_gff_segment_ids=params.reference_route.segment_ids,
                                                              logger=self.logger)
                state.handle_negative_velocities(self.logger)

            self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, state))

            # Longitudinal planning horizon (Ts)
            T_target_horizon = params.target_time - state.ego_state.timestamp_in_sec
            T_trajectory_end_horizon = max(TRAJECTORY_NUM_POINTS * TRAJECTORY_TIME_RESOLUTION,
                                           params.trajectory_end_time - state.ego_state.timestamp_in_sec)

            self.logger.debug("input: target_state: %s", params.target_state)
            self.logger.debug("input: reference_route[0]: %s", params.reference_route.points[0])
            self.logger.debug("input: ego: pos: (x: %f y: %f)", state.ego_state.x, state.ego_state.y)
            self.logger.debug("input: ego: velocity: %s", state.ego_state.velocity)
            self.logger.debug("TrajectoryPlanningFacade is required to plan with time horizon = %s", T_target_horizon)
            self.logger.debug("state: %d objects detected", len(state.dynamic_objects))

            control_status = self._get_current_control_status()
            is_engaged = control_status is not None and control_status.is_av_engaged()

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.

            # TODO: this check should include is_engaged with <and> *after* the LocalizationUtils check
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                updated_state = LocalizationUtils.get_state_with_expected_ego(
                    state, self._last_trajectory, self.logger, self.__class__.__name__, params.reference_route)
            else:
                updated_state = state

            MetricLogger.get_logger().bind(bp_time=params.bp_time)

            # plan a trajectory according to specification from upper DM level
            with DMProfiler(self.__class__.__name__ + '.plan'):
                samplable_trajectory, ctrajectories, _ = self._strategy_handlers[params.strategy].plan(
                    updated_state, params.reference_route, params.target_state, T_target_horizon,
                    T_trajectory_end_horizon, params.cost_params)

            trajectory_msg = self.generate_trajectory_plan(timestamp=state.ego_state.timestamp_in_sec,
                                                           creation_time=scene_dynamic.s_Data.s_DataCreationTime,
                                                           physical_time=scene_dynamic.s_Data.s_PhysicalEventTime,
                                                           samplable_trajectory=samplable_trajectory, is_engaged=is_engaged)

            self._publish_trajectory(trajectory_msg)
            self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, trajectory_msg)

            # publish visualization/debug data - based on short term prediction aligned state!
            debug_results = TrajectoryPlanningFacade._prepare_visualization_msg(
                state, ctrajectories, max(T_target_horizon, T_trajectory_end_horizon),
                self._strategy_handlers[params.strategy].predictor, params.reference_route)

            self._publish_debug(debug_results)

            self.logger.info("%s %s", LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, time.time() - start_time)
            MetricLogger.get_logger().report()

        except StateHasNotArrivedYet:
            self.logger.warning("StateHasNotArrivedYet was raised. skipping planning.")

        except MsgDeserializationError:
            self.logger.warning("TrajectoryPlanningFacade: MsgDeserializationError was raised. skipping planning. %s ",
                                traceback.format_exc())

        # TODO - we need to handle this as an emergency.
        except CartesianLimitsViolated:
            self.logger.critical("TrajectoryPlanningFacade: NoValidTrajectoriesFound was raised. skipping planning. %s",
                                 traceback.format_exc())

        except Exception:
            self.logger.critical("TrajectoryPlanningFacade: UNHANDLED EXCEPTION in trajectory planning: %s",
                                 traceback.format_exc())

    # TODO: add map_origin that is sent from the outside
    @prof.ProfileFunction()
    def generate_trajectory_plan(self, timestamp: float, creation_time: Timestamp, physical_time: Timestamp,
                                 samplable_trajectory: SamplableTrajectory, is_engaged: bool):
        """
        sample trajectory points from the samplable-trajectory, translate them according to ego's reference point and
        wrap them in a message to the controller
        :param timestamp: the timestamp to use as a reference for the beginning of trajectory
        :param samplable_trajectory: the trajectory plan to sample points from (samplable object)
        :param is_engaged: whether AV is currently engaged
        :return: a TrajectoryPlan message ready to send to the controller
        """
        trajectory_num_points = min(int(np.floor(samplable_trajectory.T_extended / TRAJECTORY_TIME_RESOLUTION)),
                                    MAX_TRAJECTORY_WAYPOINTS)
        trajectory_points = samplable_trajectory.sample(
            np.linspace(start=0,
                        stop=(trajectory_num_points - 1) * TRAJECTORY_TIME_RESOLUTION,
                        num=trajectory_num_points) + timestamp)

        # TODO: this should be set with: if is_engaged else None
        self._last_trajectory = samplable_trajectory

        # publish results to the lower DM level (Control)
        # TODO: put real values in tolerance and maximal velocity fields
        # TODO: understand if padding with zeros is necessary
        allowed_tracking_errors = np.ones(shape=[trajectory_num_points, 4]) * [REPLANNING_LAT,  # left
                                                                               REPLANNING_LAT,  # right
                                                                               REPLANNING_LON,  # front
                                                                               REPLANNING_LON]  # rear
        waypoints = np.hstack((trajectory_points, allowed_tracking_errors,
                               np.zeros(shape=[trajectory_num_points, TRAJECTORY_WAYPOINT_SIZE -
                                               trajectory_points.shape[1] - allowed_tracking_errors.shape[1]])))

        timestamp_object = Timestamp.from_seconds(timestamp)
        map_origin = MapOrigin(e_phi_latitude=0, e_phi_longitude=0, e_l_altitude=0, s_Timestamp=timestamp_object)

        trajectory_plan = TrajectoryPlan(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,
                                                         e_Cnt_version=0),
                                         s_Data=DataTrajectoryPlan(s_Timestamp=timestamp_object,
                                                                   creation_time=creation_time,
                                                                   physical_time=physical_time,
                                                                   s_MapOrigin=map_origin,
                                                                   a_TrajectoryWaypoints=waypoints,
                                                                   e_Cnt_NumValidTrajectoryWaypoints=trajectory_num_points))

        return trajectory_plan

    def _validate_strategy_handlers(self) -> None:
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not self._strategy_handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(self._strategy_handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    def _get_current_scene_dynamic(self) -> SceneDynamic:
        is_success, serialized_scene_dynamic = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_DYNAMIC)

        if serialized_scene_dynamic is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_DYNAMIC)
        scene_dynamic = SceneDynamic.deserialize(serialized_scene_dynamic)
        if scene_dynamic.s_Data.s_host_localization.e_Cnt_host_hypothesis_count == 0:
            raise MsgDeserializationError("SceneDynamic was received without any host localization")
        self.logger.debug(
            "%s: %f" % (LOG_MSG_SCENE_DYNAMIC_RECEIVED, scene_dynamic.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_dynamic

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        is_success, serialized_params = self.pubsub.get_latest_sample(topic=UC_SYSTEM_TRAJECTORY_PARAMS)
        if serialized_params is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed' %
                                          UC_SYSTEM_TRAJECTORY_PARAMS)
        trajectory_params = TrajectoryParams.deserialize(serialized_params)
        self.logger.debug('%s: %s', LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, trajectory_params)
        return trajectory_params

    def _publish_trajectory(self, results: TrajectoryPlan) -> None:
        self.pubsub.publish(UC_SYSTEM_TRAJECTORY_PLAN, results.serialize())

    def _publish_debug(self, debug_msg: TrajectoryVisualizationMsg) -> None:
        self.pubsub.publish(UC_SYSTEM_TRAJECTORY_VISUALIZATION, debug_msg.serialize())

    @staticmethod
    @prof.ProfileFunction()
    def _prepare_visualization_msg(state: State, ctrajectories: CartesianTrajectories,
                                   planning_horizon: float, predictor: EgoAwarePredictor,
                                   reference_route: GeneralizedFrenetSerretFrame) -> TrajectoryVisualizationMsg:
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
        alternative_ids_skip_range = np.round(
            np.linspace(0, len(ctrajectories) - 1, MAX_VIS_TRAJECTORIES_NUMBER)).astype(int)
        # slice alternative trajectories by skipping indices - for visualization
        sliced_ctrajectories = ctrajectories[alternative_ids_skip_range]

        # this assumes the state is already aligned by short time prediction
        prediction_horizons = np.arange(0, planning_horizon, VISUALIZATION_PREDICTION_RESOLUTION, float)

        # visualize objects' predictions
        # TODO: create 3 GFFs in TP and convert objects' predictions on them
        objects_visualizations = []
        for obj in state.dynamic_objects:
            try:
                obj_fstate = reference_route.cstate_to_fstate(obj.cartesian_state)
                obj_fpredictions = predictor.predict_2d_frenet_states(np.array([obj_fstate]),
                                                                      prediction_horizons)[0][:, [FS_SX, FS_DX]]
                # skip objects having predictions out of reference_route
                valid_obj_fpredictions = obj_fpredictions[obj_fpredictions[:, FP_SX] < reference_route.s_max]
                if len(valid_obj_fpredictions) == 0:
                    continue
                obj_cpredictions = reference_route.fpoints_to_cpoints(valid_obj_fpredictions)
                objects_visualizations.append(PredictionsVisualization(obj.obj_id, obj_cpredictions))

            except Exception:  # verify the object can be projected on reference_route
                continue

        header = Header(0, Timestamp.from_seconds(state.ego_state.timestamp_in_sec), 0)
        trajectory_length = ctrajectories.shape[1]
        points_step = int(trajectory_length / MAX_NUM_POINTS_FOR_VIZ) + 1
        visualization_data = DataTrajectoryVisualization(
            sliced_ctrajectories[:, :trajectory_length:points_step, :(C_Y + 1)],
            # at most MAX_NUM_POINTS_FOR_VIZ points
            objects_visualizations, "")
        return TrajectoryVisualizationMsg(header, visualization_data)

    def _get_current_control_status(self) -> Optional[ControlStatus]:
        is_success, serialized_control_status = self.pubsub.get_latest_sample(topic=UC_SYSTEM_CONTROL_STATUS)

        if serialized_control_status is None:
            # this is a warning and not an assert, since currently perfect_control does not publish the message
            self.logger.warning("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                UC_SYSTEM_CONTROL_STATUS)
            return None
        control_status = ControlStatus.deserialize(serialized_control_status)
        self.logger.debug("%s: %f engaged %s" % (LOG_MSG_CONTROL_STATUS, control_status.s_Header.s_Timestamp.timestamp_in_seconds,
                                                 control_status.s_Data.e_b_TTCEnabled))
        return control_status


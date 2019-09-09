import time
import traceback
from logging import Logger

import numpy as np
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_ROUTE_PLAN
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_DYNAMIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TAKEOVER
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_VISUALIZATION
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException, StateHasNotArrivedYet, \
    RepeatedRoadSegments, EgoRoadSegmentNotFound, EgoStationBeyondLaneLength, EgoLaneOccupancyCostIncorrect, \
    RoutePlanningException, MappingException, raises
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, BEHAVIORAL_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED, \
    MIN_DISTANCE_TO_SET_TAKEOVER_FLAG, TIME_THRESHOLD_TO_SET_TAKEOVER_FLAG, LOG_MSG_SCENE_DYNAMIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.takeover_message import Takeover, DataTakeover
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import State, EgoState
from decision_making.src.utils.dm_profiler import DMProfiler
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.utils.metric_logger.metric_logger import MetricLogger


class BehavioralPlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, behavioral_planner: CostBasedBehavioralPlanner,
                 last_trajectory: SamplableTrajectory = None) -> None:
        """
        :param pubsub:
        :param logger:
        :param behavioral_planner:
        :param last_trajectory: last trajectory returned from behavioral planner.
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self._planner = behavioral_planner
        self.logger.info("Initialized Behavioral Planner Facade.")
        self._last_trajectory = last_trajectory
        self._last_gff_segment_ids = np.array([])
        self._started_receiving_states = False
        MetricLogger.init(BEHAVIORAL_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        self.pubsub.subscribe(UC_SYSTEM_SCENE_DYNAMIC)
        self.pubsub.subscribe(UC_SYSTEM_SCENE_STATIC)
        self.pubsub.subscribe(UC_SYSTEM_ROUTE_PLAN)

    def _stop_impl(self):
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_DYNAMIC)
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_STATIC)
        self.pubsub.unsubscribe(UC_SYSTEM_ROUTE_PLAN)

    def _periodic_action_impl(self) -> None:
        """
        The main function of the behavioral planner. It read the most up-to-date state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """

        try:
            start_time = time.time()

            with DMProfiler(self.__class__.__name__ + '._get_current_route_plan'):
                route_plan = self._get_current_route_plan()
                route_plan_dict = route_plan.to_costs_dict()

            with DMProfiler(self.__class__.__name__ + '.get_scene_static'):
                scene_static = self._get_current_scene_static()
                SceneStaticModel.get_instance().set_scene_static(scene_static)

            with DMProfiler(self.__class__.__name__ + '._get_current_scene_dynamic'):
                scene_dynamic = self._get_current_scene_dynamic()
                state = State.create_state_from_scene_dynamic(scene_dynamic=scene_dynamic,
                                                              selected_gff_segment_ids=self._last_gff_segment_ids,
                                                              route_plan_dict=route_plan_dict,
                                                              logger=self.logger)

                state.handle_negative_velocities(self.logger)

            if scene_dynamic.s_Data.s_host_localization.e_Cnt_host_hypothesis_count > 1:
                self.logger.debug("Multiple localization hypotheses published for ego vehicle at timestamp: %f," +
                                  " Number of Hypotheses: %d",
                                  state.ego_state.timestamp_in_sec,
                                  scene_dynamic.s_Data.s_host_localization.e_Cnt_host_hypothesis_count)

            self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, state))

            # Tests if actual localization is close enough to desired localization, and if it is, it starts planning
            # from the DESIRED localization rather than the ACTUAL one. This is due to the nature of planning with
            # Optimal Control and the fact it complies with Bellman principle of optimality.
            # THIS DOES NOT ACCOUNT FOR: yaw, velocities, accelerations, etc. Only to location.
            if LocalizationUtils.is_actual_state_close_to_expected_state(
                    state.ego_state, self._last_trajectory, self.logger, self.__class__.__name__):
                updated_state = self._get_state_with_expected_ego(state)
                self.logger.debug("BehavioralPlanningFacade ego localization was overridden to the expected-state "
                                  "according to previous plan")
            else:
                updated_state = state

            # calculate the takeover message
            takeover_message = self._set_takeover_message(route_plan_data=route_plan.s_Data, ego_state=updated_state.ego_state)

            self._publish_takeover(takeover_message)

            with DMProfiler(self.__class__.__name__ + '.plan'):
                trajectory_params, samplable_trajectory, behavioral_visualization_message = self._planner.plan(updated_state,
                                                                                                           route_plan)

            self._last_trajectory = samplable_trajectory

            self._last_gff_segment_ids = trajectory_params.reference_route.segment_ids

            # Send plan to trajectory
            self._publish_results(trajectory_params)

            # Send visualization data
            self._publish_visualization(behavioral_visualization_message)

            self.logger.info("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, time.time() - start_time))

            MetricLogger.get_logger().report()

        except StateHasNotArrivedYet:
            self.logger.warning("StateHasNotArrivedYet was raised. skipping planning.")

        except MsgDeserializationError as e:
            self.logger.warning("MsgDeserializationError was raised. skipping planning. " +
                                "turn on debug logging level for more details.%s" % (traceback.format_exc()))
            self.logger.debug(str(e))

        except BehavioralPlanningException as e:
            self.logger.warning(e)

        except RoutePlanningException as e:
            self.logger.warning(e)

        except MappingException as e:
            self.logger.warning(e)

        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s" %
                                 (e, traceback.format_exc()))

    def _get_current_route_plan(self) -> RoutePlan:
        """
        Returns the last received route plan data
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized RoutePlan
        """
        is_success, serialized_route_plan = self.pubsub.get_latest_sample(topic=UC_SYSTEM_ROUTE_PLAN)
        if serialized_route_plan is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_ROUTE_PLAN)
        route_plan = RoutePlan.deserialize(serialized_route_plan)
        self.logger.debug("Received route plan: %s" % route_plan)
        return route_plan

    @raises(EgoRoadSegmentNotFound, RepeatedRoadSegments, EgoStationBeyondLaneLength, EgoLaneOccupancyCostIncorrect)
    def _set_takeover_message(self, route_plan_data:DataRoutePlan, ego_state:EgoState) -> Takeover:
        """
        Calculate the takeover message based on the static route plan
        The takeover flag will be set to True if all lane end costs for a downstream road segment
        within a threshold distance are 1 (i.e. road is blocked).
        :param route_plan_data: last route plan data
        :param ego_state: last state for ego vehicle
        :return: Takeover data
        """

        ego_lane_segment_id = ego_state.map_state.lane_id

        ego_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(ego_lane_segment_id)

        # find ego road segment row index in as_route_plan_lane_segments 2-d array which matches the index in a_i_road_segment_ids 1-d array
        route_plan_start_idx = np.argwhere(route_plan_data.a_i_road_segment_ids == ego_road_segment_id)

        if route_plan_start_idx.size == 0:  # check if ego road segment Id is listed inside route plan data
            raise EgoRoadSegmentNotFound('Route plan does not include data for ego road segment ID {0}'.format(ego_road_segment_id))
        elif route_plan_start_idx.size > 1:
            raise RepeatedRoadSegments('Route Plan has repeated data for road segment ID {0}'.format(ego_road_segment_id))

        ego_row_idx = route_plan_start_idx[0][0]

        dist_to_end = 0.0
        takeover_flag = False

        # iterate through all road segments within MIN_DISTANCE_TO_SET_TAKEOVER_FLAG
        for route_row_idx in range(ego_row_idx, route_plan_data.e_Cnt_num_road_segments):

            # raise exception if ego lane occupancy cost is 1
            if route_row_idx == ego_row_idx:
                ego_road_lane_ids = np.array([route_lane.e_i_lane_segment_id for route_lane in route_plan_data.as_route_plan_lane_segments[ego_row_idx]])
                ego_col_idx = np.argwhere(ego_road_lane_ids == ego_lane_segment_id)[0][0]
                if route_plan_data.as_route_plan_lane_segments[ego_row_idx][ego_col_idx].e_cst_lane_occupancy_cost == 1:
                    raise EgoLaneOccupancyCostIncorrect('Occupancy cost is 1 for ego lane segment ID {0}'.format(ego_lane_segment_id))

            # find the length of the road segment, assuming that road segment length is similar
            # to its first lane segment length
            road_segment_lane_id = route_plan_data.as_route_plan_lane_segments[route_row_idx][0].e_i_lane_segment_id
            lane = MapUtils.get_lane(road_segment_lane_id)
            lane_length = lane.e_l_length

            if route_row_idx == ego_row_idx:
                dist_to_end = lane_length - ego_state.map_state.lane_fstate[FS_SX]
            else:
                dist_to_end += lane_length

            if dist_to_end >= max(MIN_DISTANCE_TO_SET_TAKEOVER_FLAG, ego_state.map_state.lane_fstate[FS_SV] * TIME_THRESHOLD_TO_SET_TAKEOVER_FLAG):
                # If the host is far from any potential problem area, break the loop. The takeover flag should not be raised.
                break
            else:
                # Since the host is close to a potential problem area, check the end cost for all lane segments within the road segment. If all of the
                # lane end costs are equal to 1, there is no where for the host to go. Set the takeover flag to True and break the loop.
                lane_end_costs = np.array([route_lane.e_cst_lane_end_cost for route_lane in route_plan_data.as_route_plan_lane_segments[route_row_idx]])

                if np.all(lane_end_costs == 1):
                    takeover_flag = True
                    break

        takeover_message = Takeover(s_Header=Header(e_Cnt_SeqNum=0,
                                                    s_Timestamp=Timestamp.from_seconds(ego_state.timestamp_in_sec),
                                                    e_Cnt_version=0),
                                    s_Data=DataTakeover(takeover_flag))

        return takeover_message

    def _get_current_scene_static(self) -> SceneStatic:
        with DMProfiler(self.__class__.__name__ + '.get_latest_sample'):
            is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC)

        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        with DMProfiler(self.__class__.__name__ + '.deserialize'):
            scene_static = SceneStatic.deserialize(serialized_scene_static)
        if scene_static.s_Data.s_SceneStaticBase.e_Cnt_num_lane_segments == 0 and scene_static.s_Data.s_SceneStaticBase.e_Cnt_num_road_segments == 0:
            raise MsgDeserializationError("SceneStatic map was received without any road or lanes")
        self.logger.debug("%s: %f" % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    def _get_current_scene_dynamic(self) -> SceneDynamic:
        is_success, serialized_scene_dynamic = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_DYNAMIC)

        if serialized_scene_dynamic is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_DYNAMIC)
        scene_dynamic = SceneDynamic.deserialize(serialized_scene_dynamic)
        if scene_dynamic.s_Data.s_host_localization.e_Cnt_host_hypothesis_count == 0:
            raise MsgDeserializationError("SceneDynamic was received without any host localization")
        self.logger.debug("%s: %f" % (LOG_MSG_SCENE_DYNAMIC_RECEIVED, scene_dynamic.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_dynamic

    def _get_state_with_expected_ego(self, state: State) -> State:
        """
        takes a state and overrides its ego vehicle's localization to be the localization expected at the state's
        timestamp according to the last trajectory cached in the facade's self._last_trajectory.
        Note: lateral velocity is zeroed since we don't plan for drifts and lateral components are being reflected in
        yaw and curvature.
        :param state: the state to process
        :return: a new state object with a new ego-vehicle localization
        """
        current_time = state.ego_state.timestamp_in_sec
        expected_state_vec: CartesianExtendedState = self._last_trajectory.sample(np.array([current_time]))[0]
        expected_ego_state = state.ego_state.clone_from_cartesian_state(expected_state_vec, state.ego_state.timestamp_in_sec)

        updated_state = state.clone_with(ego_state=expected_ego_state)
        # mark this state as a state which has been sampled from a trajectory and wasn't received from state module
        updated_state.is_sampled = True

        return updated_state

    def _publish_results(self, trajectory_parameters: TrajectoryParams) -> None:
        self.pubsub.publish(UC_SYSTEM_TRAJECTORY_PARAMS, trajectory_parameters.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, trajectory_parameters))

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.pubsub.publish(UC_SYSTEM_VISUALIZATION, visualization_message.serialize())

    def _publish_takeover(self, takeover_message:Takeover) -> None :
        self.pubsub.publish(UC_SYSTEM_TAKEOVER, takeover_message.serialize())

    @property
    def planner(self):
        return self._planner

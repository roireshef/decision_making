import time
import traceback
from logging import Logger

import numpy as np

from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_STATE
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_NAVIGATION_PLAN
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_VISUALIZATION
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_ROUTE_PLAN
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TAKEOVER

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException, StateHasNotArrivedYet
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, BEHAVIORAL_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED , DISTANCE_TO_SET_TAKEOVER_FLAG
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.state.state import State
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.scene.scene_static_model import SceneStaticModel
import rte.python.profiler as prof

from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.takeover_message import Takeover, DataTakeover
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.planning.types import C_Y, FS_SX


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
        self._started_receiving_states = False
        MetricLogger.init(BEHAVIORAL_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        self.pubsub.subscribe(UC_SYSTEM_STATE)
        self.pubsub.subscribe(UC_SYSTEM_NAVIGATION_PLAN)
        self.pubsub.subscribe(UC_SYSTEM_SCENE_STATIC)
        self.pubsub.subscribe(UC_SYSTEM_ROUTE_PLAN)

    # TODO: unsubscribe once logic is fixed in LCM
    def _stop_impl(self):
        pass

    def _periodic_action_impl(self) -> None:
        """
        The main function of the behavioral planner. It read the most up-to-date state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """

        try:
            start_time = time.time()
            state = self._get_current_state()


            scene_static = self._get_current_scene_static()
            SceneStaticModel.get_instance().set_scene_static(scene_static)

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

            navigation_plan = self._get_current_navigation_plan()

            # get current route plan
            route_plan = self._get_current_route_plan()
            # calculate the takeover message
            takeover_msg = self.set_takeover_message(route_plan , updated_state)
            # publish takeover message
            self._publish_takeover(takeover_msg)


            trajectory_params, samplable_trajectory, behavioral_visualization_message = self._planner.plan(updated_state, navigation_plan)

            self._last_trajectory = samplable_trajectory

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

        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s" %
                                 (e, traceback.format_exc()))

    def _get_current_state(self) -> State:
        """
        Returns the last received world state.
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized State
        """
        is_success, serialized_state = self.pubsub.get_latest_sample(topic=UC_SYSTEM_STATE, timeout=1)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_state is None:
            if self._started_receiving_states:
                # PubSub queue is empty after being non-empty for a while
                raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_STATE)
            else:
                # Pubsub queue is empty since planning module is up
                raise StateHasNotArrivedYet("Waiting for data from SceneProvider/StateModule")
        self._started_receiving_states = True
        state = State.deserialize(serialized_state)
        self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, state))
        return state

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        is_success, serialized_nav_plan = self.pubsub.get_latest_sample(topic=UC_SYSTEM_NAVIGATION_PLAN, timeout=1)
        if serialized_nav_plan is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_NAVIGATION_PLAN)
        nav_plan = NavigationPlanMsg.deserialize(serialized_nav_plan)
        self.logger.debug("Received navigation plan: %s" % nav_plan)
        return nav_plan

    def _get_current_route_plan(self) -> RoutePlan:
        is_success, input_route_plan = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_ROUTE_PLAN"], timeout=1)
        object_route_plan = RoutePlan.deserialize(input_route_plan)
        self.logger.debug("Received route plan: %s" % object_route_plan)
        return object_route_plan

    @staticmethod
    def set_takeover_message(route_plan:RoutePlan, state:State ) -> Takeover:

        # find current lane segment ID
        ego_lane_id = MapUtils.get_closest_lane(state.ego_state.cartesian_state[:(C_Y+1)])
        # find current road segment ID
        curr_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(ego_lane_id)
        # find road segment index in route plan 2-d array
        route_plan_idx = [i for i in range(route_plan.s_Data.e_Cnt_num_road_segments) \
                            if route_plan.s_Data.a_i_road_segment_ids[i]==curr_road_segment_id ]

        assert(len(route_plan_idx)==1 and route_plan_idx[0] >= 0 and route_plan_idx[0] < route_plan.s_Data.e_Cnt_num_road_segments )

        row_idx = route_plan_idx[0]

         # find station on the current lane
        ego_station = state.ego_state.map_state[FS_SX]
        #find length of the lane segment
        ego_lane_length = MapUtils.get_lane_length(ego_lane_id)
        # distance to the end of current road (lane) segment
        dist_to_end = ego_lane_length - ego_station

        # check the end costs for the current road segment lanes
        blockage_flag = True
        for i in range(row_idx,route_plan.s_Data.e_Cnt_num_road_segments):

            for j in range(route_plan.s_Data.a_Cnt_num_lane_segments[i]) :
                if route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost < 1 :
                    blockage_flag = False
                    break

            # check how many road segments are within the horizon
            if i > row_idx:
                next_road_lane_id = route_plan.s_Data.as_route_plan_lane_segments[i][0].e_i_lane_segment_id
                lane_length = MapUtils.get_lane_length(next_road_lane_id)
                dist_to_end += lane_length
            if dist_to_end >= DISTANCE_TO_SET_TAKEOVER_FLAG :
                break

        if blockage_flag == True and dist_to_end < DISTANCE_TO_SET_TAKEOVER_FLAG:
            takeover_flag = True

        # TODO check this timestamp
        timestamp_object = Timestamp.from_seconds(state.ego_state.timestamp_in_sec)

        takeover_msg = Takeover(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,e_Cnt_version=0) , \
                                s_Data = DataTakeover(takeover_flag) )

        return takeover_msg


    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC, timeout=1)

        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        if scene_static.s_Data.e_Cnt_num_lane_segments == 0 and scene_static.s_Data.e_Cnt_num_road_segments == 0:
            raise MsgDeserializationError("SceneStatic map was received without any road or lanes")
        self.logger.debug("%s: %f" % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

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

    def _publish_takeover(self, takeover_msg:Takeover) -> None :
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TAKEOVER"], takeover_msg.serialize())

    @property
    def planner(self):
        return self._planner

 #   @property
 #   def predictor(self):
 #       return self._predictor

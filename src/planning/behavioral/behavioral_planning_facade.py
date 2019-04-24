import time
import traceback
from decision_making.src.utils.map_utils import MapUtils
from logging import Logger

import numpy as np

from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_STATE_LCM
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_NAVIGATION_PLAN_LCM
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS_LCM
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_VISUALIZATION_LCM

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException, StateHasNotArrivedYet
from decision_making.src.global_constants import LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, BEHAVIORAL_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_static_message import SceneStatic, StaticTrafficFlowControl, RoadObjectType
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import \
    CostBasedBehavioralPlanner
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.state.state import State
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.scene.scene_static_model import SceneStaticModel
import rte.python.profiler as prof


def patch_scene_static(lane_id, s):
    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=s,
                                         e_Pct_confidence=1.0)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(stop_sign)



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
        self.pubsub.subscribe(UC_SYSTEM_STATE_LCM, None)
        self.pubsub.subscribe(UC_SYSTEM_NAVIGATION_PLAN_LCM, None)
        self.pubsub.subscribe(UC_SYSTEM_SCENE_STATIC, None)

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

            patch_scene_static(58369795, 75)

            #MapUtils.get_lane_frenet_frame(state.ego_state.map_state.lane_id).fpoint_to_cpoint([75,0])



            print(f'* located at lane:{state.ego_state.map_state.lane_id}'
                  f' with s:{state.ego_state.map_state.lane_fstate[0]} and'
                  f' beyond_stop_bar: {state.ego_state.map_state.lane_fstate[0]>75}')


            with prof.time_range('BP-IF'):
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
        with prof.time_range('_get_current_state.get_latest_sample'):
            is_success, serialized_state = self.pubsub.get_latest_sample(topic=UC_SYSTEM_STATE_LCM, timeout=1)

        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_state is None:
            if self._started_receiving_states:
                # PubSub queue is empty after being non-empty for a while
                raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_STATE_LCM)
            else:
                # Pubsub queue is empty since planning module is up
                raise StateHasNotArrivedYet("Waiting for data from SceneProvider/StateModule")
        self._started_receiving_states = True
        state = State.deserialize(serialized_state)
        self.logger.debug('{}: {}'.format(LOG_MSG_RECEIVED_STATE, state))
        return state

    def _get_current_navigation_plan(self) -> NavigationPlanMsg:
        with prof.time_range('_get_current_navigation_plan.get_latest_sample'):
            is_success, serialized_nav_plan = self.pubsub.get_latest_sample(topic=UC_SYSTEM_NAVIGATION_PLAN_LCM, timeout=1)

        if serialized_nav_plan is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_NAVIGATION_PLAN_LCM)
        nav_plan = NavigationPlanMsg.deserialize(serialized_nav_plan)
        self.logger.debug("Received navigation plan: %s" % nav_plan)
        return nav_plan

    def _get_current_scene_static(self) -> SceneStatic:
        with prof.time_range('_get_current_scene_static.get_latest_sample'):
            is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC, timeout=1)

        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        with prof.time_range('_get_current_scene_static.SceneStatic.deserialize'):
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
        self.pubsub.publish(UC_SYSTEM_TRAJECTORY_PARAMS_LCM, trajectory_parameters.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, trajectory_parameters))

    def _publish_visualization(self, visualization_message: BehavioralVisualizationMsg) -> None:
        self.pubsub.publish(UC_SYSTEM_VISUALIZATION_LCM, visualization_message.serialize())

    @property
    def planner(self):
        return self._planner

    @property
    def predictor(self):
        return self._predictor

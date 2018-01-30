from typing import Dict

import numpy as np

from common_data.lcm.config import pubsub_topics
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.log_analysis.log_messages import LogMsg
from decision_making.test.log_analysis.parse_log_messages import LOG_PATH_FOR_ANALYSIS, STATE_IDENTIFIER_STRING_BP, \
    STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger


# TODO: Remove temporary TP facade. used only to bypass the Lcm Ser/Deser methods
class TrajectoryPlanningFacadeNoLcm(TrajectoryPlanningFacade):
    def _get_current_state(self) -> State:
        """
        Returns the last received world state.
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized State
        """
        input_state = self.pubsub.get_latest_sample(topic=pubsub_topics.STATE_TOPIC, timeout=1)
        object_state = LogMsg.deserialize(class_type=State, message=input_state)
        self.logger.debug('Received state: %s' % object_state)
        return object_state

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        input_params = self.pubsub.get_latest_sample(topic=pubsub_topics.TRAJECTORY_PARAMS_TOPIC, timeout=1)
        object_params = LogMsg.deserialize(class_type=TrajectoryParams, message=input_params)
        self.logger.debug('Received mission params: {}'.format(object_params))
        return object_params


def execute_tp(state_serialized: Dict, tp_params_serialized: Dict) -> None:
    """
    Executes the Trajectory planner with the relevant inputs by sending them via PubSub mock
    :param state_serialized: serialized state input message
    :param tp_params_serialized: serialized trajectory parameters input message
    :return:
    """

    # Create PubSub Mock
    pubsub = PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))

    # Publish messages using pubsub mock
    pubsub.publish(pubsub_topics.STATE_TOPIC, state_serialized)
    pubsub.publish(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, tp_params_serialized)

    # Initialize TP
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    MapService.initialize()
    predictor = RoadFollowingPredictor(logger)
    planner = WerlingPlanner(logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
    trajectory_planning_module = TrajectoryPlanningFacadeNoLcm(pubsub=pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers,
                                                          short_time_predictor=predictor)

    # Execute TP
    trajectory_planning_module._periodic_action_impl()


if __name__ == '__main__':
    filename = LOG_PATH_FOR_ANALYSIS
    f = open(file=filename, mode='r')
    log_content = f.readlines()

    ###########################
    # Parse relevant messages
    ###########################

    # State module
    state_module_log_timestamp, state_module_timestamps, state_module_states = DmLogParser.parse_state_message(
        log_content=log_content,
        identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE)

    # BP module
    bp_state_log_timestamp, bp_state_timestamps, bp_states = DmLogParser.parse_state_message(log_content=log_content,
                                                                                             identifier_str=STATE_IDENTIFIER_STRING_BP)
    bp_module_log_timestamp, bp_module_timestamps, bp_module_states = DmLogParser.parse_bp_output(
        log_content=log_content)

    # TP module
    tp_state_log_timestamp, tp_state_timestamps, tp_states = DmLogParser.parse_state_message(log_content=log_content,
                                                                                             identifier_str=STATE_IDENTIFIER_STRING_TP)
    tp_module_log_timestamp, tp_module_timestamps, tp_module_states = DmLogParser.parse_tp_params(
        log_content=log_content)
    no_valid_trajectories_log_timestamp, no_valid_trajectories_timestamps, no_valid_trajectories_states = DmLogParser.parse_no_valid_trajectories_message(
        log_content=log_content)

    ###########################
    # Find index of relevant messages
    ###########################
    # target_log_time = 57653.6 # Time where no valid trajectories were found
    target_log_time = 57652.6  # Time where with valid trajectories
    tp_message_index = np.where(tp_module_log_timestamp > target_log_time)[0][0]
    state_message_index = np.where(state_module_log_timestamp > target_log_time)[0][0]

    ###########################
    # Send messages to module
    ###########################

    # Convert log messages to dict
    tp_params_msg = LogMsg.convert_message_to_dict(tp_module_states[tp_message_index])
    state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])

    # Deserialize from dict to object
    tp_params = LogMsg.deserialize(class_type=TrajectoryParams, message=tp_params_msg)
    state = LogMsg.deserialize(class_type=State, message=state_msg)

    # Serialize object to PubSub dict
    state_serialized = state.to_dict()
    tp_params_serialized = tp_params.to_dict()

    ###########################
    # Execute TP with relevant inputs
    ###########################
    execute_tp(state_serialized=state_serialized, tp_params_serialized=tp_params_serialized)

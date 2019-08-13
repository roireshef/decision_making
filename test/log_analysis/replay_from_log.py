import sys

from decision_making.paths import Paths
from decision_making.src.scene.scene_static_model import SceneStaticModel
from typing import Dict

import numpy as np
import pickle

from interface.Rte_Types.python.uc_system import UC_SYSTEM_STATE
from interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING, PG_PICKLE_FILE_NAME
from decision_making.src.messages.class_serialization import ClassSerializer
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.log_analysis.parse_log_messages import STATE_IDENTIFIER_STRING_BP, \
    STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from rte.python.logger.AV_logger import AV_Logger

LOG_PATH_FOR_ANALYSIS = '/home/xzjsyy/av_code/spav/logs/AV_Log_dm_main.log'
TARGET_LOG_TIME = 57906.0


# TODO: Remove temporary TP facade. used only to bypass the Lcm Ser/Deser methods
class TrajectoryPlanningFacadeNoLcm(TrajectoryPlanningFacade):

    def _get_current_state(self) -> State:
        """
        Returns the last received world state.
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized State
        """
        input_state = self.pubsub.get_latest_sample(topic=UC_SYSTEM_STATE)
        object_state = ClassSerializer.deserialize(class_type=State, message=input_state)
        return object_state

    def _get_mission_params(self) -> TrajectoryParams:
        """
        Returns the last received mission (trajectory) parameters.
        We assume that if no updates have been received since the last call,
        then we will output the last received trajectory parameters.
        :return: deserialized trajectory parameters
        """
        input_params = self.pubsub.get_latest_sample(topic=UC_SYSTEM_TRAJECTORY_PARAMS)
        object_params = ClassSerializer.deserialize(class_type=TrajectoryParams, message=input_params)
        return object_params


def execute_tp(state_serialized: Dict, tp_params_serialized: Dict) -> None:
    """
    Executes the Trajectory planner with the relevant inputs by sending them via PubSub mock
    :param state_serialized: serialized state input message
    :param tp_params_serialized: serialized trajectory parameters input message
    :return:
    """
    scene_static_pg_no_split = pickle.load(open(Paths.get_scene_static_absolute_path_filename(PG_PICKLE_FILE_NAME), 'rb'))
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_no_split)

    # Create PubSub Mock
    pubsub = PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))

    # Publish messages using pubsub mock
    pubsub.publish(UC_SYSTEM_STATE, state_serialized)
    pubsub.publish(UC_SYSTEM_TRAJECTORY_PARAMS, tp_params_serialized)

    # Initialize TP
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    predictor = RoadFollowingPredictor(logger)
    planner = WerlingPlanner(logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
    trajectory_planning_module = TrajectoryPlanningFacadeNoLcm(pubsub=pubsub, logger=logger,
                                                               strategy_handlers=strategy_handlers)

    # Execute TP
    trajectory_planning_module._periodic_action_impl()


def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']
    target_log_time = TARGET_LOG_TIME if ('log_time' not in sys.argv) else sys.argv['log_time']

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
    no_valid_trajectories_log_timestamp = DmLogParser.parse_no_valid_trajectories_message(log_content=log_content)

    ###########################
    # Find index of relevant messages
    ###########################
    tp_params_message_index = np.where(tp_module_log_timestamp <= target_log_time)[0][-1]
    tp_state_message_index = np.where(tp_state_log_timestamp <= target_log_time)[0][-1]

    ###########################
    # Send messages to module
    ###########################

    # Convert log messages to dict
    tp_params_msg = ClassSerializer.convert_message_to_dict(tp_module_states[tp_params_message_index])
    tp_state_msg = ClassSerializer.convert_message_to_dict(tp_states[tp_state_message_index])

    # Deserialize from dict to object
    tp_params = ClassSerializer.deserialize(class_type=TrajectoryParams, message=tp_params_msg)
    tp_state = ClassSerializer.deserialize(class_type=State, message=tp_state_msg)

    # Serialize object to PubSub dict
    tp_state_serialized = tp_state.to_dict()
    tp_params_serialized = tp_params.to_dict()

    ###########################
    # Execute TP with relevant inputs
    ###########################
    try:
        execute_tp(state_serialized=tp_state_serialized, tp_params_serialized=tp_params_serialized)
        assert True

    except Exception as e:
        assert False


if __name__ == '__main__':
    main()

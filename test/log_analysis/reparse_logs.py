import sys
from typing import Dict
import json
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
from decision_making.test.log_analysis.parse_log_messages import STATE_IDENTIFIER_STRING_BP, \
    STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

# LOG_PATH_FOR_ANALYSIS = '/home/max/AV_Log_dm_main_test-2017_12_12-10_19.log'
LOG_PATH_FOR_ANALYSIS = '/data/recordings/cdrive/Database/2018_02_19/2018_02_19_17_08_Proving_Grounds_-_Low_Light/AV_Log_dm_main.log'
TARGET_LOG_TIME = 57906.0


def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']

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
    # Send messages to module
    ###########################

    with open("av_log_states.json", 'w') as file:
        for state_message_index in range(len(state_module_states)):
             # Convert log messages to dict
            state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])

            # Deserialize from dict to object
            state = LogMsg.deserialize(class_type=State, message=state_msg)
            state_serialized = state.to_dict()
            state_serialized['msg_type'] = "state_output"
            state_serialized['log_timestamp'] = state_module_log_timestamp[state_message_index]
            file.write(json.dumps(state_serialized))
            file.write("\n")
            # for object1 in tp_params.dynamic_objects:
            #     file.write(str(object1.timestamp)+","+str(object1.obj_id)+","+str(object1.v_x)+","+str(object1.v_y)+","+str(tp_params.ego_state.v_x)+","+str(tp_params.ego_state.timestamp))
            #     file.write("\n")

        # for tp_params_message_index in range(len(tp_module_states)):
        #     # Convert log messages to dict
        #     tp_params_msg = LogMsg.convert_message_to_dict(tp_module_states[tp_params_message_index])
        #
        #     # Deserialize from dict to object
        #     tp_params = LogMsg.deserialize(class_type=TrajectoryParams, message=tp_params_msg)
        #     tp_params_serialized = tp_params.to_dict()
        #     tp_params_serialized['msg_type'] = "tp_input_params"
        #     tp_params_serialized['log_timestamp'] = tp_module_log_timestamp[tp_params_message_index]
        #     file.write(json.dumps(tp_params_serialized))
        #     file.write("\n")
        #
        # for tp_state_message_index in range(len(tp_states)):
        #     tp_state_msg = LogMsg.convert_message_to_dict(tp_states[tp_state_message_index])
        #     tp_state = LogMsg.deserialize(class_type=State, message=tp_state_msg)
        #
        #     # Serialize object to PubSub dict
        #     tp_state_serialized = tp_state.to_dict()
        #     tp_state_serialized['msg_type'] = "tp_input_state"
        #     tp_state_serialized['log_timestamp'] = tp_state_log_timestamp[tp_state_message_index]
        #     file.write(json.dumps(tp_state_serialized))
        #     file.write("\n")
        #
        # for bp_message_index in range(len(bp_states)):
        #     bp_state_msg = LogMsg.convert_message_to_dict(bp_states[bp_message_index])
        #     bp_state = LogMsg.deserialize(class_type=State, message=bp_state_msg)
        #
        #     # Serialize object to PubSub dict
        #     bp_state_serialized = bp_state.to_dict()
        #     bp_state_serialized['msg_type'] = "bp_input_state"
        #     bp_state_serialized['log_timestamp'] = bp_state_log_timestamp[bp_message_index]
        #     file.write(json.dumps(bp_state_serialized))
        #     file.write("\n")
        #
        # for bp_params_message_index in range(len(bp_module_states)):
        #     # Convert log messages to dict
        #     bp_params_msg = LogMsg.convert_message_to_dict(bp_module_states[bp_params_message_index])
        #
        #     # Deserialize from dict to object
        #     bp_params = LogMsg.deserialize(class_type=TrajectoryParams, message=bp_params_msg)
        #     bp_params_serialized = bp_params.to_dict()
        #     bp_params_serialized['msg_type'] = "bp_output_params"
        #     bp_params_serialized['log_timestamp'] = bp_module_log_timestamp[bp_params_message_index]
        #     file.write(json.dumps(bp_params_serialized))
        #     file.write("\n")


if __name__ == '__main__':
    main()
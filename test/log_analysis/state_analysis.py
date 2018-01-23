import ast
import re

import matplotlib.pyplot as plt
import numpy as np

from common_data.lcm.config import pubsub_topics
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.log_messages import LogTypedMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

LOG_TIME_PATTERN = ": \d+-\d+-\d+ \d+:\d+:\d+,\d+ :"
LOG_TIME_PARSE_PATTERN = ": (\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+),(\d+) :"


def parse_state_struct(str):
    state_match = re.match(pattern='[^{]*({.*})$', string=str)
    return state_match.groups()[0]

    # return timestamp in [sec]
    return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0


def parse_log_timestamp(str):
    time_match = re.match(pattern=LOG_TIME_PARSE_PATTERN, string=str)
    y, mm, d, h, m, s, ms = float(time_match.groups()[0]), float(time_match.groups()[1]), float(
        time_match.groups()[2]), float(time_match.groups()[3]), float(time_match.groups()[4]), float(
        time_match.groups()[5]), float(time_match.groups()[6])

    # return timestamp in [sec]
    return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0


def parse_state(log_content, identifier_str):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(parse_state_struct(str=state_match.groups()[2]))
            timestamp_match = re.match(pattern=".*'timestamp': (\d*)", string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0]) * 1E-9
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_tp(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "_get_mission_params : Received mission params: "
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(parse_state_struct(str=state_match.groups()[2]))
            timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0])
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_bp(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "BehavioralPlanningFacade output is "
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[2])
            # timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
            # timestamp = float(timestamp_match.groups()[0])
            timestamp = 0.0
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_no_valid_trajectories(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "No valid trajectories found."
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[2])
            timestamp_match = re.match(pattern=".*'ego_state': {'obj_id': \d*, 'timestamp': (\d*)",
                                       string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0]) * 1E-9
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


if __name__ == '__main__':

    pubsub = PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))
    filename = '/data/recordings/cdrive/Database/2017_12_27/logs/16_02/AV_Log_dm_main_test.log'
    # filename = 'C:/Users/xzjsyy/av_code/AV_Log_dm_main_test.log'

    f = open(file=filename, mode='r')
    log_content = f.readlines()

    # State module
    identifier_str = "_publish_state_if_full: publishing state "
    state_module_log_timestamp, state_module_timestamps, state_module_states = parse_state(log_content=log_content,
                                                                                           identifier_str=identifier_str)

    # BP module
    identifier_str = "behavioral_facade.py:    76: _get_current_state  : Received State: "
    bp_state_log_timestamp, bp_state_timestamps, bp_states = parse_state(log_content=log_content,
                                                                         identifier_str=identifier_str)
    bp_module_log_timestamp, bp_module_timestamps, bp_module_states = parse_bp(log_content=log_content)

    # TP module
    identifier_str = "trajectory_planning_facade.py:   139: _get_current_state  : Received state: "
    tp_state_log_timestamp, tp_state_timestamps, tp_states = parse_state(log_content=log_content,
                                                                         identifier_str=identifier_str)
    tp_module_log_timestamp, tp_module_timestamps, tp_module_states = parse_tp(log_content=log_content)
    no_valid_trajectories_log_timestamp, no_valid_trajectories_timestamps, no_valid_trajectories_states = parse_no_valid_trajectories(
        log_content=log_content)


    # Find index where no trajectories were found
    invalid_log_time = 57653.6
    invalid_tp_message_index = np.where(tp_module_log_timestamp > invalid_log_time)[0][0]
    invalid_state_message_index = np.where(state_module_log_timestamp > invalid_log_time)[0][0]

    # Find index where trajectories were found
    valid_log_time = 57652.6
    valid_tp_message_index = np.where(tp_module_log_timestamp > valid_log_time)[0][0]
    valid_state_message_index = np.where(state_module_log_timestamp > valid_log_time)[0][0]

    # Fetch messages from log
    tp_params_dict_1 = ast.literal_eval(tp_module_states[invalid_tp_message_index])
    state_dict_1 = ast.literal_eval(state_module_states[invalid_state_message_index])
    tp_params_1 = LogTypedMsg.deserialize(class_type=TrajectoryParams, message=tp_params_dict_1)
    state_1 = LogTypedMsg.deserialize(class_type=State, message=state_dict_1)

    # Publish messages using pubsub mock
    pubsub.publish(pubsub_topics.STATE_TOPIC, LogTypedMsg.serialize(state_1))
    pubsub.publish(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, LogTypedMsg.serialize(tp_params_1))

    # Initialize TP
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    MapService.initialize()
    predictor = RoadFollowingPredictor(logger)
    planner = WerlingPlanner(logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
    trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)

    # Execute TP
    trajectory_planning_module._periodic_action_impl()


    # Plot log analysis
    fig = plt.figure()
    plt.plot(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
    plt.plot(state_module_log_timestamp[1:], np.diff(state_module_timestamps), '-.r')
    plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_timestamps), '-g')
    plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_timestamps), '-c')
    plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_timestamps), '-k')
    plt.title('State time measurements')
    plt.legend(('Times where no valid trajectories were found',
                'Diff between last ego timestamp in state module',
                'Diff between last ego timestamp in BP module',
                'Diff between last ego timestamp in TP module',
                'Diff between end time of high-level action'))
    plt.xlabel('Log time [sec]')
    plt.show()

    fig = plt.figure()
    plt.plot(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
    plt.plot(state_module_log_timestamp[1:], np.diff(state_module_log_timestamp), '-.r')
    plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_log_timestamp), '-g')
    plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_log_timestamp), '-c')
    plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_log_timestamp), '-k')
    plt.plot(bp_module_log_timestamp[1:], np.diff(bp_module_log_timestamp), '-m')
    plt.title('Log time measurements')
    plt.legend(('Times where no valid trajectories were found',
                'Time since last state message in state module',
                'Time since last state received in BP',
                'Time since last state received in TP',
                'Time since last TP fetch of TP params',
                'Time since last output send by BP'))
    plt.xlabel('Log time [sec]')
    plt.show()

    a = 2

import sys
import numpy as np
import matplotlib.pyplot as plt

from decision_making.test.log_analysis.parse_log_messages import DmLogParser, \
    STATE_IDENTIFIER_STRING_BP, STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE

#LOG_PATH_FOR_ANALYSIS = '/home/max/av_code/spav/logs/AV_Log_dm_main-2017_12_27_16_18.log'
#LOG_PATH_FOR_ANALYSIS = '/home/max/AV_Log_dm_main_test-2017_12_12-10_19.log'
LOG_PATH_FOR_ANALYSIS = '/home/xzjsyy/av_code/spav/logs/AV_Log_dm_main.log'
#LOG_PATH_FOR_ANALYSIS = '/data/recordings/cdrive/Database/2017_12_27/logs/16_18/AV_Log_dm_main_test.log'

def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']
    f = open(file=filename, mode='r')
    log_content = f.readlines()

    ###########################
    # Load relevant messages
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

    # Control Errors
    control_errors_log_timestamp, control_errors_lon, control_errors_lat = DmLogParser.parse_control_error(
        log_content=log_content)

    ###########################
    # Plot log analysis
    ###########################

    # Plot messages by message timestamp
    fig = plt.subplot(3, 1, 1)
    if no_valid_trajectories_log_timestamp.shape[0] > 0:
        plt.stem(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_log_timestamp.shape))
    plt.plot(state_module_log_timestamp[1:], np.diff(state_module_timestamps), '-.r')
    plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_timestamps), '-g')
    plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_timestamps), '-c')
    plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_timestamps), '-k')
    plt.title('State time measurements (%s)' % LOG_PATH_FOR_ANALYSIS)
    plt.legend(('Diff between last ego timestamp in state module',
                'Diff between last ego timestamp in BP module',
                'Diff between last ego timestamp in TP module',
                'Diff between end time of high-level action',
                'Times where no valid trajectories were found'))
    plt.xlabel('Log time [sec]')
    # plt.show()

    # Plot messages by log timestamp
    fig = plt.subplot(3, 1, 2)
    if no_valid_trajectories_log_timestamp.shape[0] > 0:
        plt.stem(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_log_timestamp.shape))
    plt.plot(state_module_log_timestamp[1:], np.diff(state_module_log_timestamp), '-.r')
    plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_log_timestamp), '-g')
    plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_log_timestamp), '-c')
    plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_log_timestamp), '-k')
    plt.plot(bp_module_log_timestamp[1:], np.diff(bp_module_log_timestamp), '-m')

    plt.title('Measurements in Log time (%s)' % LOG_PATH_FOR_ANALYSIS)
    plt.legend(('Time since last state message in state module',
                'Time since last state received in BP',
                'Time since last state received in TP',
                'Time since last TP fetch of TP params',
                'Time since last output send by BP',
                'Times where no valid trajectories were found'))
    plt.xlabel('Log time [sec]')

    # Control Errors
    fig = plt.subplot(3, 1, 3)
    if no_valid_trajectories_log_timestamp.shape[0] > 0:
        plt.stem(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_log_timestamp.shape))
    plt.plot(state_module_log_timestamp[1:], np.diff(state_module_log_timestamp), '-.r')
    plt.plot(control_errors_log_timestamp, control_errors_lon, '-.b')
    plt.plot(control_errors_log_timestamp, control_errors_lat, '-.r')
    plt.title('Control errors (%s)' % LOG_PATH_FOR_ANALYSIS)
    plt.legend(('Time since last state message in state module',
                'Control Errors (lon) [m]',
                'Control Errors (lat) [m]',
                'Times where no valid trajectories were found'))
    plt.xlabel('Log time [sec]')
    plt.ylabel('Error [m]')

    plt.show()

if __name__ == '__main__':
    main()
import sys

from decision_making.src.state.state import State
from decision_making.test.log_analysis.log_messages import LogMsg
from decision_making.test.log_analysis.parse_log_messages import STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser



def main(log_filename: str = None, state_identifier_string: str = STATE_IDENTIFIER_STRING_STATE_MODULE):


    #if no filename had been passed as argument, either command-line or function call, then use a hard coded local file.
    if log_filename is None:
        if len(sys.argv) < 2:
            log_filename = '/home/vzt9fl/av_code/spav/logs/AV_Log_dm_main.log'
        else:
            log_filename = sys.argv[1]

    f = open(file=log_filename, mode='r')

    log_content = f.readlines()
    f.close()

    _, _, state_module_states = DmLogParser.parse_state_message(
        log_content=log_content,
        identifier_str=state_identifier_string)

    out_file=open('state_dump.txt', 'w')
    for single_state_log in state_module_states:
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(single_state_log)
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        out_file.write(str(state.ego_state) + '\n')

    out_file.close()

if __name__ == '__main__':
    main()
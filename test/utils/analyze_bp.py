from decision_making.paths import Paths
import matplotlib.pyplot as plt

BP_TIME_TEXT = 'BehavioralFacade._periodic_action_impl'
CRITICAL = 'CRITICAL'
TP_BP_TIME = 'TP_BP_EGO_TIME'
BP_TIME = ' BP_EGO_TIME'
TP_TIME = ' TP_EGO_TIME'
STATE_TIME = 'STATE_EGO_TIME'
STATE_S = 'STATE_S'
STATE_EXEC_TIME = 'STATE_EXEC_TIME'


def get_time(time_string):
    hours, minutes, seconds = time_string.split(':')
    return float(hours)*60*60 + float(minutes)*60 + float(seconds.split(',')[0]) + float(seconds.split(',')[1])/1000

def get_single_measurement(file, match_string):
    with open(file, 'r') as f:
        return [(get_time(line.split()[3]), float(line.split()[14])) for line in f if line.find(match_string) != -1]

def get_bp_exec_times(file):
    with open(file, 'r') as f:
        return [(get_time(line.split()[3]), float(line.split()[15])) for line in f if line.find(BP_TIME_TEXT) != -1]

def get_exceptions(file):
    with open(file, 'r') as f:
        return [(get_time(line.split()[3]), line.split()[13]) for line in f if line.find(CRITICAL) != -1]


if __name__ == "__main__":
    file = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    bp_times = get_bp_exec_times(file)
    bp_ego_times = get_single_measurement(file, BP_TIME)
    tp_ego_times = get_single_measurement(file, TP_TIME)
    tp_bp_ego_times = get_single_measurement(file, TP_BP_TIME)
    state_ego_times = get_single_measurement(file, STATE_TIME)
    state_exec_times = get_single_measurement(file, STATE_EXEC_TIME)
    state_s = get_single_measurement(file, STATE_S)
    print(state_exec_times)

    exceptions = get_exceptions(file)

    #plt.scatter([t for t,b in bp_ego_times],[b for t,b in bp_ego_times], c='blue')
    #plt.scatter([t for t,b in tp_ego_times],[b for t,b in tp_ego_times], c='green')
    #plt.scatter([t for t,b in tp_bp_ego_times],[b for t,b in tp_bp_ego_times], facecolors='none', edgecolors='r')
    #plt.scatter([t for t,b in state_ego_times],[b/1e8 for t,b in state_ego_times], facecolors='none', edgecolors='black')
    #plt.plot([t for t,b in state_exec_times],[b for t,b in state_exec_times])
    #plt.scatter([t for t,b in state_s],[b for t,b in state_s], facecolors='none', edgecolors='purple')

    plt.plot([t for t,b in bp_times],[b for t,b in bp_times], c='blue')
    #plt.plot([t for t,b in bp_times], [0.3]*len(bp_times), c='orange')
    #plt.scatter([t for t,e in exceptions],[0.3]*len(exceptions), c='red')
    for t,e in exceptions:
        plt.axvline(t,c='red', linestyle='--' , lw='.1')

    plt.show()



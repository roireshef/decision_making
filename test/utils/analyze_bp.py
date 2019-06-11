from collections import defaultdict

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
PROF = 'PROF'

FILE_NAME = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()

def get_profs():
    data=defaultdict(list)
    with open(FILE_NAME, 'r') as f:
        for line in f:
            if line.find(PROF) == -1:
                continue
            time = get_time(line.split()[3])
            label, instance = line.split()[13].split(':')
            data[label].append((time, float(instance)))
    return data




def get_time(time_string):
    hours, minutes, seconds = time_string.split(':')
    return float(hours)*60*60 + float(minutes)*60 + float(seconds.split(',')[0]) + float(seconds.split(',')[1])/1000

def get_single_measurement(match_string):
    with open(FILE_NAME, 'r') as f:
        return [(get_time(line.split()[3]), float(line.split()[14])) for line in f if line.find(match_string) != -1]

def get_bp_exec_times():
    with open(FILE_NAME, 'r') as f:
        return [(get_time(line.split()[3]), float(line.split()[15])) for line in f if line.find(BP_TIME_TEXT) != -1]

def get_exceptions():
    with open(FILE_NAME, 'r') as f:
        return [(get_time(line.split()[3]), line.split()[13]) for line in f if line.find(CRITICAL) != -1]

def scatter_timed_series(timed_series, **kwargs):
    times, measurements = zip(*timed_series)
    plt.scatter(times, measurements, **kwargs)

def plot_timed_series(timed_series, **kwargs):
    times, measurements = zip(*timed_series)
    plt.plot(times, measurements, **kwargs)

def plot_timed_series_labeled(timed_series, label, **kwargs):
    times, measurements = zip(*timed_series)
    plt.plot(times, measurements, label=label, **kwargs)


def plot_exceptions():
    exceptions = get_exceptions()
    for t,e in exceptions:
        plt.axvline(t, c='red', linestyle='--' , lw='.1')


def plot_bp_execution_times():
    bp_times = get_bp_exec_times()
    plot_timed_series(bp_times, c='blue')
    plot_timed_series([(t, 0.3) for t, _ in bp_times], c='orange')


def plot_gap_in_scene_dynamic():
    scatter_timed_series(get_single_measurement(STATE_TIME), facecolors='none', edgecolors='black')


def plot_ego_timestamps():
    scatter_timed_series(get_single_measurement(BP_TIME))
    scatter_timed_series(get_single_measurement(TP_TIME))
    scatter_timed_series(get_single_measurement(TP_BP_TIME))
    scatter_timed_series(get_single_measurement(STATE_TIME))

def plot_profiler():
    profs = get_profs()
    for p,timed_series in profs.items():
        plot_timed_series_labeled(timed_series, p)
    plt.legend()

def summarize_profiler():
    summary = {}
    profs = get_profs()
    for p, timed_series in profs.items():
        cumtime = sum([t for _,t in timed_series])
        summary[p] = {'n_calls': len(timed_series), 'avg_time': cumtime/len(timed_series) , 'cumtime':cumtime}
    print('name      n_calls     avg_time(sec)      cumtime')
    print('---------------------------------------------------')
    for m in sorted(summary.keys(), key=lambda x: summary[x]['avg_time'], reverse=True):
        print(f"{m}     {summary[m]['n_calls']}     {summary[m]['avg_time']}    {summary[m]['cumtime']}")
    print('---------------------------------------------------')


if __name__ == "__main__":
    #summarize_profiler()
    plot_profiler()
    #print(get_profs())
    #plot_bp_execution_times()
    #plot_gap_in_scene_dynamic()
    #plot_exceptions()
    #plt.legend()
    plt.show()



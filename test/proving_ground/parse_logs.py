import ast
from decision_making.src.planning.types import FS_SV, C_V
import matplotlib.pyplot as plt
from decision_making.src.state.state import EgoState


def plot_dynamics(path: str):

    f = open(path, 'r')
    ego_cv = []
    ego_sv = []
    other_cv = []
    other_sv = []
    ego_lane = []
    other_lane = []
    timestamp_in_sec = []

    cnt = 0

    while True:
        text = f.readline()
        if not text:
            break
        if '_scene_dynamic_callback' in text:
            state_str = text.split('Publishing State ')[1]
            try:
                state_dict = ast.literal_eval(state_str)
            except ValueError as e:
                cnt += 1
                continue
            ego_cv.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_V])
            ego_sv.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SV])
            ego_lane.append(state_dict['ego_state']['_cached_map_state']['lane_id']/1e9)
            timestamp_in_sec.append(EgoState.ticks_to_sec(state_dict['ego_state']['timestamp']))

            dyn_obj_list = state_dict['dynamic_objects']['iterable']
            if len(dyn_obj_list) == 0:
                other_cv.append(0.0)
                other_sv.append(0.0)
                other_lane.append(0.0)
            else:
                other_cv.append(dyn_obj_list[0]['_cached_cartesian_state']['array'][C_V])
                other_sv.append(dyn_obj_list[0]['_cached_map_state']['lane_fstate']['array'][FS_SV])
                other_lane.append(dyn_obj_list[0]['_cached_map_state']['lane_id']/1e8)

    f = plt.figure(1)
    ego_sv_plot,  = plt.plot(timestamp_in_sec, ego_sv)
    other_sv_plot,  = plt.plot(timestamp_in_sec, other_sv)
    ego_lane_plot,  = plt.plot(timestamp_in_sec, ego_lane)
    other_lane_plot,  = plt.plot(timestamp_in_sec, other_lane)
    plt.xlabel('time[s]')
    plt.ylabel('velocity[m/s]')
    plt.legend([ego_sv_plot, other_sv_plot, ego_lane_plot, other_lane_plot], ['ego_sv', 'other_sv', 'ego_lane', 'other_lane'])
    f.show()

    g = plt.figure(2)
    ego_cv_plot,  = plt.plot(timestamp_in_sec, ego_cv)
    other_cv_plot,  = plt.plot(timestamp_in_sec, other_cv)
    plt.xlabel('time[s]')
    plt.ylabel('velocity[m/s]')
    plt.legend([ego_cv_plot, other_cv_plot], ['ego_cv', 'other_cv'])
    plt.show()


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    plot_dynamics('/home/yzystl/projects/uc_workspace/ultracruise/logs/AV_Log_dm_main.log')
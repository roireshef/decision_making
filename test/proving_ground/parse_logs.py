import ast
import matplotlib.pyplot as plt

from decision_making.paths import Paths
from decision_making.src.planning.types import FS_SV, C_V, FS_SX
from decision_making.src.state.state import EgoState


def plot_dynamics(path: str):

    f = open(path, 'r')
    ego_cv = []
    ego_sv = []
    ego_sx = []
    other_cv = []
    other_sv = []
    other_sx = []
    ego_lane = []
    other_lane = []
    timestamp_in_sec = []

    spec_t = []
    spec_v = []
    spec_s = []
    spec_time = []

    recipe_desc = []
    recipe_time = []

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
            ego_sx.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SX])
            ego_lane.append(state_dict['ego_state']['_cached_map_state']['lane_id']/1e9)
            timestamp_in_sec.append(EgoState.ticks_to_sec(state_dict['ego_state']['timestamp']))

            dyn_obj_list = state_dict['dynamic_objects']['iterable']
            if len(dyn_obj_list) == 0:
                other_cv.append(0.0)
                other_sv.append(0.0)
                other_sx.append(0.0)
                other_lane.append(0.0)
            else:
                other_cv.append(dyn_obj_list[0]['_cached_cartesian_state']['array'][C_V])
                other_sv.append(dyn_obj_list[0]['_cached_map_state']['lane_fstate']['array'][FS_SV])
                other_sx.append(dyn_obj_list[0]['_cached_map_state']['lane_fstate']['array'][FS_SX])
                other_lane.append(dyn_obj_list[0]['_cached_map_state']['lane_id']/1e9)

        if 'Chosen behavioral action spec' in text:
            spec_str = text.split('Chosen behavioral action spec ')[1]
            spec_dict = ast.literal_eval(spec_str.split(' (ego_timestamp: ')[0])

            time = float(spec_str.split(' (ego_timestamp: ')[1][:-2])

            spec_t.append(float(spec_dict['t']))
            spec_v.append(float(spec_dict['v']))
            spec_s.append(float(spec_dict['s']))
            spec_time.append(float(time))

        if 'Chosen behavioral action recipe' in text:
            recipe_str = text.split('Chosen behavioral action recipe')[1].split('Recipe: ')[1].replace("<", "'<").replace(">", ">'")
            recipe_dict = ast.literal_eval(recipe_str.split(' (ego_timestamp: ')[0])

            time = float(recipe_str.split(' (ego_timestamp: ')[1][:-2])

            recipe_desc.append('%s\n%s' % (recipe_dict['action_type'], recipe_dict['aggressiveness']))

            recipe_time.append(float(time))

    f = plt.figure(1)
    plt.plot(411)
    ax1 = plt.subplot(4, 1, 1)
    ego_sv_plot,  = plt.plot(timestamp_in_sec, ego_sv)
    other_sv_plot,  = plt.plot(timestamp_in_sec, other_sv)
    ego_lane_plot,  = plt.plot(timestamp_in_sec, ego_lane)
    other_lane_plot,  = plt.plot(timestamp_in_sec, other_lane)
    plt.xlabel('time[s]')
    plt.ylabel('velocity[m/s]')
    plt.legend([ego_sv_plot, other_sv_plot, ego_lane_plot, other_lane_plot], ['ego_sv', 'other_sv', 'ego_lane', 'other_lane'])

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ego_cx_plot,  = plt.plot(timestamp_in_sec, ego_sx)
    other_cx_plot,  = plt.plot(timestamp_in_sec, other_sx)
    plt.xlabel('time[s]')
    plt.ylabel('longitude[m]')
    plt.legend([ego_cx_plot, other_cx_plot], ['ego_s', 'other_s'])

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    plt.plot(recipe_time, recipe_desc, 'o--')
    plt.xlabel('time[s]')
    plt.ylabel('recipe')

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    spec_t_plot,  = plt.plot(spec_time, spec_t, 'o-')
    spec_v_plot,  = plt.plot(spec_time, spec_v, 'o-')
    # spec_s_plot,  = plt.plot(spec_time, spec_s)

    plt.xlabel('time[s]')
    plt.ylabel('sepc_time/spec_velocity')
    plt.legend([spec_t_plot, spec_v_plot], ['spec_t [s]', 'spec_v [m/s]'])

    plt.show()


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    plot_dynamics(file)
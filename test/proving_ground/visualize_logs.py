import ast
import matplotlib.pyplot as plt
import numpy as np

from decision_making.paths import Paths
from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LAT, NEGLIGIBLE_DISPOSITION_LON, EPS
from decision_making.src.messages.scene_common_messages import Timestamp
from decision_making.src.planning.types import FS_SV, C_V, FS_SX, FS_SA, C_A, C_K, C_X, C_Y
from decision_making.src.state.state import EgoState


def plot_dynamics(path: str):
    f = open(path, 'r')
    ego_cv = []
    ego_ca = []
    ego_curv = []
    ego_sa = []
    ego_sv = []
    ego_sx = []
    other_cv = []
    other_sv = []
    other_sx = []
    timestamp_in_sec = []

    spec_t = []
    spec_v = []
    spec_s = []
    spec_time = []

    recipe_desc = []
    recipe_time = []

    cnt = 0

    bp_if_lon_err = []
    bp_if_lat_err = []
    bp_if_time = []

    tp_if_lon_err = []
    tp_if_lat_err = []
    tp_if_time = []

    trajectory = []
    trajectory_time = []

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
            ego_ca.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_A])
            ego_curv.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_K])
            ego_sa.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SA])
            ego_sv.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SV])
            ego_sx.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SX])
            timestamp_in_sec.append(EgoState.ticks_to_sec(state_dict['ego_state']['timestamp']))

            dyn_obj_list = state_dict['dynamic_objects']['iterable']
            if len(dyn_obj_list) == 0:
                other_cv.append(0.0)
                other_sv.append(0.0)
                other_sx.append(0.0)
            else:
                other_cv.append(dyn_obj_list[0]['_cached_cartesian_state']['array'][C_V])
                other_sv.append(dyn_obj_list[0]['_cached_map_state']['lane_fstate']['array'][FS_SV])
                other_sx.append(dyn_obj_list[0]['_cached_map_state']['lane_fstate']['array'][FS_SX])

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

        if 'is_actual_state_close_to_expected_state stats called from' in text:
            if_str = text.split('is_actual_state_close_to_expected_state stats called from ')
            dict_start = if_str[1].index('{')
            dict_str = if_str[1][dict_start:]
            kv_strings = dict_str[1:-2].split(', ')
            errors = np.fromstring(kv_strings[4].split(': ')[1][1:-1], sep=' ')

            if 'BehavioralPlanningFacade' in if_str[1]:
                bp_if_lon_err.append(errors[0])
                bp_if_lat_err.append(errors[1])
                bp_if_time.append(float(kv_strings[-1].split(': ')[1]))

            if 'TrajectoryPlanningFacade' in if_str[1]:
                tp_if_lon_err.append(errors[0])
                tp_if_lat_err.append(errors[1])
                tp_if_time.append(float(kv_strings[-1].split(': ')[1]))

        if 'Publishing Trajectory' in text:
            traj_str = text.split('Publishing Trajectory: ')[1]
            traj_dict = ast.literal_eval(traj_str)
            points_counter = int(traj_dict['s_Data']['e_Cnt_NumValidTrajectoryWaypoints'])
            timestamp = Timestamp(traj_dict['s_Data']['s_Timestamp']['e_Cnt_Secs'], traj_dict['s_Data']['s_Timestamp']['e_Cnt_FractionSecs'])
            trajectory.append(np.array(traj_dict['s_Data']['a_TrajectoryWaypoints']['array']).reshape(
                traj_dict['s_Data']['a_TrajectoryWaypoints']['shape'])[:points_counter])
            trajectory_time.append(timestamp.timestamp_in_seconds)

    f = plt.figure(1)

    ax1 = plt.subplot(5, 2, 1)
    ego_cx_plot,  = plt.plot(timestamp_in_sec, ego_sx)
    other_cx_plot,  = plt.plot(timestamp_in_sec, other_sx)
    plt.xlabel('time[s]')
    plt.ylabel('longitude[m]')
    plt.legend([ego_cx_plot, other_cx_plot], ['ego_s', 'other_s'])

    ax2 = plt.subplot(5, 2, 3, sharex=ax1)
    ego_sv_plot,  = plt.plot(timestamp_in_sec, ego_sv)
    other_sv_plot,  = plt.plot(timestamp_in_sec, other_sv)
    plt.xlabel('time[s]')
    plt.ylabel('velocity[m/s]')
    plt.legend([ego_sv_plot, other_sv_plot], ['ego_sv', 'other_sv'])

    ax3 = plt.subplot(5, 2, 5, sharex=ax1)
    ego_sa_plot,  = plt.plot(timestamp_in_sec, ego_sa)
    plt.xlabel('time[s]')
    plt.ylabel('acceleration[m]')
    plt.legend([ego_sa_plot], ['ego_sa'])

    ax4 = plt.subplot(5, 2, 7, sharex=ax1)
    plt.plot(recipe_time, recipe_desc, 'o--')
    plt.xlabel('time[s]')
    plt.ylabel('recipe')

    ax5 = plt.subplot(5, 2, 9, sharex=ax1)
    spec_t_plot,  = plt.plot(spec_time, spec_t, 'o-')
    spec_v_plot,  = plt.plot(spec_time, spec_v, 'o-')

    plt.xlabel('time[s]')
    plt.ylabel('spec_time/spec_velocity')
    plt.legend([spec_t_plot, spec_v_plot], ['spec_t [s]', 'spec_v [m/s]'])

    ax6 = plt.subplot(5, 2, 2, sharex=ax1)
    bp_if_lon,  = plt.plot(bp_if_time, bp_if_lon_err, 'o-.')
    bp_if_lat,  = plt.plot(bp_if_time, bp_if_lat_err, 'o--')
    tp_if_lon,  = plt.plot(tp_if_time, tp_if_lon_err, 'o-.')
    tp_if_lat,  = plt.plot(tp_if_time, tp_if_lat_err, 'o--')

    lon_th = plt.axhline(y=NEGLIGIBLE_DISPOSITION_LON, linewidth=1, color='k', linestyle='-.')
    lat_th = plt.axhline(y=NEGLIGIBLE_DISPOSITION_LAT, linewidth=1, color='k', linestyle='--')

    plt.xlabel('time[s]')
    plt.ylabel('loc/tracking errors')
    plt.legend([bp_if_lon, bp_if_lat, tp_if_lon, tp_if_lat, lon_th, lat_th],
               ['BP-Lon', 'BP-Lat', 'TP-Lon', 'TP-Lat', 'Lon threshold', 'Lat threshold'])

    ax7 = plt.subplot(5, 2, 4, sharex=ax1)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_X], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('trajectories (x position)')

    ax8 = plt.subplot(5, 2, 6, sharex=ax1)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_Y], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('trajectories (y position)')

    ax9 = plt.subplot(5, 2, 8, sharex=ax1)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_V], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('trajectories (vel.)')

    ax10 = plt.subplot(5, 2, 10, sharex=ax1)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_A], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('trajectories (acc.)')

    plt.show()


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    plot_dynamics(file)
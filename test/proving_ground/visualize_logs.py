import ast

import matplotlib.pyplot as plt
import numpy as np
from decision_making.paths import Paths
from decision_making.src.global_constants import REPLANNING_LAT, REPLANNING_LON
from decision_making.src.messages.scene_common_messages import Timestamp
from decision_making.src.planning.behavioral.data_objects import ActionType, AggressivenessLevel
from decision_making.src.planning.types import FS_SV, C_V, FS_SX, FS_SA, C_A, C_K, C_X, C_Y, FS_DX
from decision_making.src.planning.utils.math_utils import Math


def plot_dynamics(log_file_path: str):
    """
    Plot various graphs concerning localization, BP and TP outputs
    :param log_file_path: path to AV_Log_dm_main.log file
    :return: a showable matplotlib figure
    """
    f = open(log_file_path, 'r')
    ego_hypothesis_num = []
    multiple_ego_hypotheses_timestamp = []
    ego_cv = []
    ego_ca = []
    ego_curv = []
    ego_sa = []
    ego_sv = []
    ego_sx = []
    ego_dx = []
    other_vels = []
    timestamp_in_sec = []

    spec_t = []
    spec_v = []
    spec_s = []
    spec_time = []

    recipe_action = []
    recipe_aggresiveness = []
    recipe_time = []

    bp_if_lon_err = []
    bp_if_lat_err = []
    bp_if_time = []

    tp_if_lon_err = []
    tp_if_lat_err = []
    tp_if_time = []

    trajectory = []
    trajectory_time = []
    no_valid_traj_timestamps = []
    no_action_in_bp_timestamps = []
    other_ids = []
    other_times = []
    other_dists = []

    min_headway_calm = []
    min_headway_std = []
    min_headway_aggr = []
    min_headway_chosen = []
    min_headway_time = []
    v_T_mod = []
    v_T_mod1 = []
    v_T_mod_time = []
    lower_root = []

    stop_dist = []
    stop_dist_timestamp = []

    ego_lane_id = None
    vel_limit = []
    vel_limit_time = []

    engaged = []
    engaged_time = []

    while True:
        text = f.readline()
        if not text:
            break

        if 'Received state' in text:
            state_str = text.split('Received state: ')[1].replace('inf', 'None')
            state_dict = ast.literal_eval(state_str)
            ego_cv.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_V])
            ego_ca.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_A])
            ego_curv.append(state_dict['ego_state']['_cached_cartesian_state']['array'][C_K])
            if state_dict['ego_state']['_cached_map_state'] is not None:
                ego_sa.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SA])
                ego_sv.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SV])
                ego_sx.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_SX])
                ego_dx.append(state_dict['ego_state']['_cached_map_state']['lane_fstate']['array'][FS_DX])
                ego_lane_id = state_dict['ego_state']['_cached_map_state']['lane_id']
            else:
                ego_sa.append(0)
                ego_sv.append(0)
                ego_sx.append(0)
                ego_dx.append(0)
            timestamp_in_sec.append(Math.ticks_to_sec(state_dict['ego_state']['timestamp']))

        if 'BehavioralGrid: time' in text:
            behavioral_grid_str = text.split('BehavioralGrid: time')[1]
            time = float(behavioral_grid_str.split(', ')[0])
            obj_dist = float(behavioral_grid_str.split('dist_from_front_object ')[1].split(',')[0])
            obj_str = behavioral_grid_str.split('front_object: ')[1]
            obj_dict = ast.literal_eval(obj_str)
            if obj_dict is not None:
                obj_id = obj_dict['obj_id']
                obj_vel = obj_dict['_cached_cartesian_state']['array'][C_V]
                other_times.append(time)
                other_ids.append(obj_id)
                other_vels.append(obj_vel)
                other_dists.append(obj_dist)

        if 'Scene Dynamic host localization published' in text:
            ego_hypothesis_num.append(int(text.split('Number of Hypotheses: ')[1].split(', Hypotheses')[0]))
            if ego_hypothesis_num[-1] > 1:
                multiple_ego_hypotheses_timestamp.append(float(text.split('at timestamp: ')[1].split(', Number of Hypotheses:')[0]))

        if 'Chosen behavioral action spec' in text:
            spec_str = text.split('Chosen behavioral action spec ')[1]
            spec_dict = ast.literal_eval(spec_str.split(' (ego_timestamp: ')[0])

            time = float(spec_str.split(' (ego_timestamp: ')[1][:-2])

            spec_t.append(float(spec_dict['t']))
            spec_v.append(float(spec_dict['v']))
            spec_s.append(float(spec_dict['s']))
            spec_time.append(float(time))

        if 'NoActionsLeftForBPError:' in text:
            no_action_in_bp_timestamps.append(float(text.split('timestamp_in_sec: ')[1]))

        if 'Chosen behavioral action recipe' in text:
            recipe_str = text.split('Chosen behavioral action recipe')[1].split('Recipe: ')[1].replace("<", "'<").replace(">", ">'")
            recipe_dict = ast.literal_eval(recipe_str.split(' (ego_timestamp: ')[0])

            time = float(recipe_str.split(' (ego_timestamp: ')[1][:-2])

            action_type = int(recipe_dict['action_type'].replace('>', '').split(':')[1])
            action_type = action_type + 2 if action_type < ActionType.OVERTAKE_VEHICLE.value else action_type + 1  # remove the OVERTAKE - not in use
            aggressiveness = int(recipe_dict['aggressiveness'].replace('>', '').split(':')[1])
            recipe_action.append(action_type)
            recipe_aggresiveness.append(aggressiveness)

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

        if 'CartesianLimitsViolated' in text:
            no_valid_traj_timestamps.append(float(text.split('timestamp_in_sec: ')[1].split(',')[0]))

        if 'Headway min' in text:
            try:
                split_str = text.split('Headway min')[1].split(',')
                min_headway_calm.append(float(split_str[0]))
                min_headway_std.append(float(split_str[1]))
                min_headway_aggr.append(float(split_str[2]))
                min_headway_chosen.append(int(split_str[3]))
                min_headway_time.append(float(split_str[4]))
            except:
                pass

        if 'SlowDown' in text:
            split_str = text.split('SlowDown')[1].split(',')
            v_T_mod1.append(float(split_str[2]))
            v_T_mod.append(float(split_str[3]))
            lower_root.append(float(split_str[4]))
            v_T_mod_time.append(float(split_str[5]))

        if 'STOP RoadSign distance:' in text:
            stop_dist.append(float(text.split('STOP RoadSign distance:')[1]))
            stop_dist_timestamp.append(time)

        if 'Speed limits at time' in text:
            if ego_lane_id is not None:
                speed_limit_per_lane = ast.literal_eval(text.split('Speed limits at time')[1].split(': ', maxsplit=1)[1])
                if ego_lane_id in list(speed_limit_per_lane):
                    vel_limit.append(speed_limit_per_lane[ego_lane_id])
                    vel_limit_time.append(float(text.split('Speed limits at time')[1].split(':')[0]))

        if 'Received ControlStatus message' in text:
            msg = text.split('Timestamp: :')[1]
            parts = msg.split('engaged')
            try:
                engaged_time.append(time)  # using time since the timestamp attached to this message is in system time, not ego time
                engaged.append(int(parts[1]))
            except NameError:
                pass  # do nothing if time was not initialized yet

    f = plt.figure(1)

    ax1 = plt.subplot(5, 2, 1)
    ego_sx_plot,  = plt.plot(timestamp_in_sec, ego_sx)
    longitudinal_dist_plot, = plt.plot(other_times, other_dists, '.-')
    stop_dist_plt, = plt.plot(stop_dist_timestamp, stop_dist)
    multiple_ego_hypotheses = plt.scatter(multiple_ego_hypotheses_timestamp, [100] * len(multiple_ego_hypotheses_timestamp), s=5, c='k')
    plt.xlabel('time[s]')
    plt.ylabel('longitude[m]/distance[m]')
    plt.legend([ego_sx_plot, longitudinal_dist_plot, multiple_ego_hypotheses, stop_dist_plt], ['ego_s', 'longitudinal_dist', 'multiple_ego_hypotheses', 'stop dist'])
    other_times, other_ids, other_dists = np.array(other_times, dtype=float), np.array(other_ids, dtype=int), np.array(other_dists, dtype=float)
    switch_idx = np.where(other_ids[:-1] - other_ids[1:] != 0)[0] + 1
    if len(switch_idx) > 0:
        switch_idx = np.concatenate(([0], switch_idx))
        plt.scatter(other_times[switch_idx], other_dists[switch_idx], marker='o', c=longitudinal_dist_plot.get_color())
        for idx in switch_idx:
            plt.annotate(str(other_ids[idx]), (other_times[idx], other_dists[idx]))
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True)

    ax2 = plt.subplot(5, 2, 3, sharex=ax1)
    ego_sv_plot,  = plt.plot(timestamp_in_sec, ego_sv)
    other_vel_plot,  = plt.plot(other_times, other_vels, '.-')
    speed_limit_plt, = plt.plot(vel_limit_time, vel_limit, '--', color='k')
    # un remark the lines below to see the way the target speed is modified
    # mod_v_t_plot, = plt.plot(v_T_mod_time, v_T_mod, 'x')
    # mod_v_t1_plot, = plt.plot(v_T_mod_time, v_T_mod1, '+')
    # lower_root_plot, = plt.plot(v_T_mod_time, lower_root, 's')
    plt.xlabel('time[s]')
    plt.ylabel('velocity[m/s]')
    plt.legend([ego_sv_plot, other_vel_plot, speed_limit_plt], ['ego_vel', 'other_vel', 'lane_limit'])
    # plt.legend([ego_sv_plot, other_sv_plot, mod_v_t_plot, mod_v_t1_plot, lower_root_plot], ['ego_sv', 'other_sv', 'other_mod', 'other_mod1', 'lower root'], loc='upper right')
    plt.grid(True)

    ax3 = plt.subplot(5, 2, 5, sharex=ax1)
    ego_sa_plot,  = plt.plot(timestamp_in_sec, ego_sa)
    ego_sj = [(sa2 - sa1) / (t2 - t1) for (sa2, sa1, t2, t1) in
              zip(ego_sa[1:], ego_sa[0:-1], timestamp_in_sec[1:], timestamp_in_sec[0:-1]) if (t2 - t1 > 0.01)]
    ego_sj_time = [t2 for (t2, t1) in zip(timestamp_in_sec[1:], timestamp_in_sec[0:-1]) if (t2 - t1 > 0.01)]
    ego_sj_plot, = plt.plot(ego_sj_time, ego_sj)
    plt.xlabel('time[s]')
    plt.ylabel('acceleration[m/s^2]')
    plt.legend([ego_sa_plot, ego_sj_plot], ['ego_sa', 'ego_sj'])
    plt.grid(True)

    ax4 = plt.subplot(5, 2, 2, sharex=ax1)
    plt.plot(recipe_time, recipe_action, '.-', color='g')
    plt.plot(recipe_time, recipe_aggresiveness, '.-', color='m')
    plt.xlabel('time[s]')
    plt.ylabel('Chosen Recipe')
    y_values = [str(aggressiveness).split('.')[1].lower() for aggressiveness in AggressivenessLevel] + \
               [str(action).split('.')[1].lower() for action in ActionType if action != ActionType.OVERTAKE_VEHICLE]
    y_axis = np.arange(len(y_values))
    plt.yticks(y_axis, y_values)
    plt.axhline(y=2.5, linewidth=1, color='k', linestyle='-')
    plt.grid(True)

    ax5 = plt.subplot(5, 2, 4, sharex=ax1)
    spec_t_plot,  = plt.plot(spec_time, spec_t, 'o-')
    spec_v_plot,  = plt.plot(spec_time, spec_v, 'o-')
    bp_no_actions_plot = plt.scatter(no_action_in_bp_timestamps, [1]*len(no_action_in_bp_timestamps), s=5, c='k')

    plt.xlabel('time[s]')
    plt.ylabel('Action Horizon\nTarget Velocity')
    plt.legend([spec_t_plot, spec_v_plot, bp_no_actions_plot], ['spec_t [s]', 'spec_v [m/s]', 'no_actions_bp'])
    plt.grid(True)

    ax6 = plt.subplot(5, 2, 9, sharex=ax1)
    bp_if_lon,  = plt.plot(bp_if_time, bp_if_lon_err, 'o-.')
    bp_if_lat,  = plt.plot(bp_if_time, bp_if_lat_err, 'o--')
    tp_if_lon,  = plt.plot(tp_if_time, tp_if_lon_err, 'o-.')
    tp_if_lat,  = plt.plot(tp_if_time, tp_if_lat_err, 'o--')
    engaged_plt, = plt.plot(engaged_time, engaged, 'o--')

    lon_th = plt.axhline(y=REPLANNING_LON, linewidth=1, color='k', linestyle='-.')
    lat_th = plt.axhline(y=REPLANNING_LAT, linewidth=1, color='k', linestyle='--')

    plt.xlabel('time[s]')
    plt.ylabel('loc/tracking errors')
    plt.legend([bp_if_lon, bp_if_lat, tp_if_lon, tp_if_lat, lon_th, lat_th, engaged_plt],
               ['BP-Lon', 'BP-Lat', 'TP-Lon', 'TP-Lat', 'Lon threshold', 'Lat threshold', 'engaged'])
    plt.grid(True)

    ax7 = plt.subplot(5, 2, 7, sharex=ax1)
    ego_curv_plt, = plt.plot(timestamp_in_sec, ego_curv)
    plt.legend([ego_curv_plt], ['curvature (cartesian)'])

    plt.xlabel('time[s]')
    plt.ylabel('curvature [1/m]')
    plt.grid(True)
    # min_headway_time = np.array(min_headway_time)
    # min_headway_calm = np.array(min_headway_calm)
    # valid_idx = np.where((min_headway_calm > 0.0) & (min_headway_calm < 5.0))[0]
    # min_headway_calm_plt, = plt.plot(min_headway_time[valid_idx], min_headway_calm[valid_idx], 'x')
    # min_headway_std = np.array(min_headway_std)
    # valid_idx = np.where((min_headway_std > 0.0) & (min_headway_calm < 5.0))[0]
    # min_headway_std_plt, = plt.plot(min_headway_time[valid_idx], 2+min_headway_std[valid_idx], 'o')
    # min_headway_aggr = np.array(min_headway_aggr)
    # valid_idx = np.where((min_headway_aggr > 0.0) & (min_headway_calm < 5.0))[0]
    # min_headway_agg_plt, = plt.plot(min_headway_time[valid_idx], 4+min_headway_aggr[valid_idx], '+')
    # min_headway_chosen = np.array(min_headway_chosen)
    # valid_idx = np.where(min_headway_chosen >= 0)[0]
    # min_headway_chosen_plt, = plt.plot(min_headway_time[valid_idx], min_headway_chosen[valid_idx], '.')
    # plt.xlabel('time[s]')
    # plt.ylabel('headway')
    # plt.legend([min_headway_calm_plt, min_headway_std_plt, min_headway_agg_plt, min_headway_chosen_plt],
    #            ['min_calm', 'min std', 'min aggr', 'chosen'])
    # plt.grid(True)

    ax8 = plt.subplot(5, 2, 6, sharex=ax1)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_X], '-.')
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_Y], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('location (x,y) requests \n(trajectory)')
    plt.grid(True)

    ax9 = plt.subplot(5, 2, 8, sharex=ax1)
    ego_sv_plt, = plt.plot(timestamp_in_sec, ego_sv, 'k-', alpha=0.2)
    plt.plot(timestamp_in_sec[0:-1:10], ego_sv[0:-1:10], 'kx', alpha=0.2)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_V], '-.')

    plt.xlabel('time[s]')
    plt.ylabel('velocity requests\n(trajectory)')
    plt.legend([ego_sv_plt], ['actual vel.'])
    plt.grid(True)

    ax10 = plt.subplot(5, 2, 10, sharex=ax1)
    ego_sa_plt, = plt.plot(timestamp_in_sec, ego_sa, 'k-', alpha=0.2)
    plt.plot(timestamp_in_sec[0:-1:10], ego_sa[0:-1:10], 'kx', alpha=0.2)
    for t, traj in zip(trajectory_time, trajectory):
        plt.plot(t + np.arange(len(traj)) * 0.1, traj[:, C_A], '-.')
    no_valid_traj_plot = plt.scatter(no_valid_traj_timestamps, [1]*len(no_valid_traj_timestamps), s=5, c='k')

    plt.xlabel('time[s]')
    plt.ylabel('acceleration requests\n(trajectory)')
    plt.legend([ego_sa_plt, no_valid_traj_plot], ['actual acc.', 'no_valid_traj_tp'])
    plt.grid(True)

    return f


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file_path = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    f = plot_dynamics(file_path)
    plt.show(f)

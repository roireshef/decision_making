from logging import Logger

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import sys

from decision_making.src.global_constants import EPS, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SX, FS_SA, FS_DA, FS_DX, C_A, C_V, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, EgoState, ObjectSize


def sample_curvature(num_samples: int) -> [np.array, np.array]:
    """
    Build a smooth road with zigzags along global X axis
    :param num_samples: number of random samples
    :return: 2D matrix of random samples; each sample contains: init velocity, end velocity, encoded road curvature
    """
    # define road parameters
    num_road_zigzags = 4                # number of zigzags of the curve road
    road_length = 100                   # length of the encoded part of the road
    zigzag_sigma = 0.023 * road_length  # zigzags' strength

    # define velocity parameters
    vel_mean = 12                       # mean velocity (normal truncated distribution)
    vel_sigma = 6                       # sigma velocity (normal truncated distribution)
    min_limit = vel_mean - 2 * vel_sigma
    max_limit = vel_mean + 2 * vel_sigma

    # define curvature encoding parameters
    curvature_pieces_num = 8            # number of segments, by which the curvature array is divided (for encoding)

    max_accel_arr = np.zeros(num_samples)
    vel_arr = np.empty((0, 2))
    curvatures_arr = np.empty((0, curvature_pieces_num))

    for i in range(num_samples):

        points_x = np.linspace(0, road_length, num=num_road_zigzags) + \
                   np.random.uniform(-0.5*road_length/num_road_zigzags, 0.5*road_length/num_road_zigzags)
        points = np.c_[points_x, np.zeros(num_road_zigzags)]
        points[:, 1] = zigzag_sigma * np.random.randn(num_road_zigzags)

        # add far point to prevent from trajectories to overflow frenet
        ext_points = np.concatenate((np.array([-road_length, 0])[np.newaxis], points, np.array([2*road_length, 0])[np.newaxis]), axis=0)

        frenet = FrenetSerret2DFrame(ext_points, ds=1)
        from_idx = np.argwhere(frenet.O[:, 0] >= 0)[0][0]
        # till_idx = np.argwhere(frenet.O[:, 0] > road_x_range)[0][0]

        #plt.subplot(2, 1, 1)
        #plt.plot(frenet.O[from_idx:till_idx, 0], frenet.O[from_idx:till_idx, 1])
        #plt.axis('equal')

        v_0 = scipy.stats.truncnorm.rvs((min_limit - vel_mean) / vel_sigma, (max_limit - vel_mean) / vel_sigma,
                                        loc=vel_mean, scale=vel_sigma, size=1)[0]
        v_T = scipy.stats.truncnorm.rvs((min_limit - vel_mean) / vel_sigma, (max_limit - vel_mean) / vel_sigma,
                                        loc=vel_mean, scale=vel_sigma, size=1)[0]
        T = min(BP_ACTION_T_LIMITS[1], 2 * road_length / (v_0 + v_T))
        vel_arr = np.concatenate((vel_arr, np.array([v_0, v_T])[np.newaxis]), axis=0)

        # map_state = MapState(init_fstate, road_id=20)
        # ego = EgoState.create_from_map_state(0, 0, map_state, ObjectSize(0,0,0), 0)
        # state = State(None, [], ego)
        # logger = Logger("test_encode_curvature")
        # behavioral_state = BehavioralGridState.create_from_state(state=state, logger=logger)
        # recipe = DynamicAction
        # action_space = ActionSpace(logger, [recipe])
        # recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)

        quartic_A_inv = np.linalg.inv(QuarticPoly1D.time_constraints_matrix(T))
        constraints_s = np.array([0, v_0, 0, v_T, 0])
        poly_coefs_s = QuarticPoly1D.solve(quartic_A_inv, constraints_s[np.newaxis, :])[0]

        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=0, T_s=T, T_d=T, frenet_frame=frenet,
                                                          poly_s_coefs=poly_coefs_s, poly_d_coefs=np.zeros(6))

        time_points = np.arange(0, T + EPS, 0.1)
        ctrajectory = samplable_trajectory.sample(time_points)
        max_lat_acc = np.max(np.abs(ctrajectory[:, C_K]) * ctrajectory[:, C_V] ** 2)
        max_accel_arr[i] = max_lat_acc

        all_curvatures = np.abs(frenet.k[from_idx:from_idx+road_length, 0])
        size = curvature_pieces_num * int(all_curvatures.shape[0] / curvature_pieces_num)
        curvatures = np.split(all_curvatures[:size], curvature_pieces_num)
        curvature_pieces = np.max(curvatures, axis=1)
        curvatures_arr = np.concatenate((curvatures_arr, curvature_pieces[np.newaxis]), axis=0)

        #plt.subplot(2, 1, 2)
        #plt.plot(frenet.k[from_idx:till_idx, 0])
        #plt.show()
        #print('%.2f %.2f %s %.2f\n' % (v_0, v_T, curvatures, max_lat_acc), end=" ")

    # print('%d' % (len(np.where(max_accel_arr < 3)[0])))

    # return np.c_[vel_arr*0.1, curvatures_arr*1000], max_accel_arr
    return np.c_[np.square(vel_arr)*0.01, np.reciprocal(curvatures_arr)*0.001], max_accel_arr

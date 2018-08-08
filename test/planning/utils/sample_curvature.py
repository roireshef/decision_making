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


# number of segments, by which the curvature array is divided (for encoding)
CURVATURE_ENCODING_PARTS_NUM = 6


def sample_curvature(num_samples: int) -> [np.array, np.array]:
    """
    Build a smooth road with zigzags along global X axis
    :param num_samples: number of random samples
    :return: 2D matrix of random samples; each sample contains: init velocity, end velocity, encoded road curvature
    """
    # define velocity parameters
    vel_mean = 24.                      # mean velocity (normal truncated distribution)
    vel_sigma = vel_mean/2              # sigma velocity (normal truncated distribution)
    min_limit = vel_mean - 2 * vel_sigma
    max_limit = vel_mean + 2 * vel_sigma

    # define road parameters
    num_road_zigzags = 4                # number of zigzags of the curve road
    road_length = vel_mean * 12         # length of the encoded part of the road
    zigzag_sigma = (5.5 / vel_mean**2) * road_length  # zigzags' strength such that half of samples have acceleration > 3

    max_accel_arr = np.zeros(num_samples)
    vel_arr = np.empty((0, 2))
    curvatures_arr = np.empty((0, CURVATURE_ENCODING_PARTS_NUM))

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

        encoded_curvature = encode_curvature(frenet.k[from_idx:from_idx+int(road_length), 0])
        curvatures_arr = np.concatenate((curvatures_arr, encoded_curvature[np.newaxis]), axis=0)

        #plt.subplot(2, 1, 2)
        #plt.plot(frenet.k[from_idx:till_idx, 0])
        #plt.show()
        #print('%.2f %.2f %s %.2f\n' % (v_0, v_T, curvatures, max_lat_acc), end=" ")

    print('%d' % (len(np.where(max_accel_arr < 3)[0])))

    # return np.c_[vel_arr*0.1, curvatures_arr*1000], max_accel_arr, np.array([0, 1, 0, 1, 0, 1])

    sq_vel = np.square(vel_arr)
    curve_radius = np.reciprocal(curvatures_arr)

    vel_normalize_factor = 2. / vel_mean**2
    curv_normalize_factor = 0.2 / vel_mean**2

    return np.c_[sq_vel * vel_normalize_factor, curve_radius * curv_normalize_factor], max_accel_arr


def encode_curvature(full_curvature: np.array) -> np.array:
    """
    Encode road's curvature array, such that it may be used as a part of encoded state representation.
    The method: split the curvature array by CURVATURE_ENCODING_PARTS_NUM and take a maximum curvature for each part.
    :param full_curvature: 1D array of road curvatures per sample point of Frenet frame.
    :return: 1D array: encoded curvature
    """
    # reduce the full_curvature array such that its size is divided by CURVATURE_ENCODING_PARTS_NUM
    aligned_array_size = full_curvature.shape[0] - full_curvature.shape[0] % CURVATURE_ENCODING_PARTS_NUM
    aligned_abs_curvature = np.abs(full_curvature[:aligned_array_size])
    split_curvatures = np.split(aligned_abs_curvature, CURVATURE_ENCODING_PARTS_NUM)
    return np.max(split_curvatures, axis=1)  # return maximum curvature in each part

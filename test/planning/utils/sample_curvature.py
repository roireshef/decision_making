from logging import Logger
from typing import List

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import sys

from decision_making.src.global_constants import EPS, BP_ACTION_T_LIMITS, LON_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SX, FS_SA, FS_DA, FS_DX, C_A, C_V, C_K, FS_SV
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, EgoState, ObjectSize


# number of segments, by which the curvature array is divided (for encoding)
from mapping.src.service.map_service import MapService

CURVATURE_ENCODING_PARTS_NUM = 8


def sample_curvature(num_samples: int) -> [np.array, np.array]:
    """
    Build a smooth road with zigzags along global X axis
    :param num_samples: number of random samples
    :return: 2D matrix of random samples; each sample contains: init velocity, end velocity, encoded road curvature
    """
    # define velocity parameters
    vel_max = 20
    vel_mean = 0.7 * vel_max            # mean velocity (normal truncated distribution)
    vel_sigma = vel_mean/4              # sigma velocity (normal truncated distribution)
    min_limit = 0
    max_limit = vel_max
    max_T = BP_ACTION_T_LIMITS[1]
    dt = 0.1

    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width

    # define road parameters
    num_road_zigzags = 4                # number of zigzags of the curve road
    road_length = vel_mean * max_T      # length of the encoded part of the road
    zigzag_amplitude = 400./vel_mean   # zigzags' strength such that half of samples have acceleration > 3

    curvatures_arr = np.empty((0, CURVATURE_ENCODING_PARTS_NUM))

    logger = Logger("sample_curvature")
    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
    same_lane_recipes = [recipe for recipe in action_space.recipes if recipe.relative_lane.value == 0]

    # initialize network ground truth
    max_accel_mat = np.zeros((num_samples, len(same_lane_recipes)))
    init_vel_arr = np.zeros(num_samples)

    map_state = MapState(road_fstate=np.array([0.01, 0, 0, lane_width/2, 0, 0]), road_id=road_id)
    ego = EgoState.create_from_map_state(0, 0, map_state, ObjectSize(4,2,0), 0)
    state = State(None, [], ego)
    time_points = np.arange(0, max_T + EPS, dt)

    for sample_idx in range(num_samples):

        points_x = np.linspace(0, road_length, num=num_road_zigzags) + \
                   np.random.uniform(-0.5*road_length/num_road_zigzags, 0.5*road_length/num_road_zigzags)
        points = np.c_[points_x, np.zeros(num_road_zigzags)]
        points[:, 1] = zigzag_amplitude * np.random.randn(num_road_zigzags)

        # add far point to prevent from trajectories to overflow frenet
        ext_points = np.concatenate((np.array([-road_length, 0])[np.newaxis], points, np.array([2*road_length, 0])[np.newaxis]), axis=0)

        frenet = FrenetSerret2DFrame(ext_points, ds=1)
        from_idx = np.argwhere(frenet.O[:, 0] >= 0)[0][0]
        # till_idx = np.argwhere(frenet.O[:, 0] > road_x_range)[0][0]

        #plt.subplot(2, 1, 1)
        #plt.plot(frenet.O[from_idx:till_idx, 0], frenet.O[from_idx:till_idx, 1])
        #plt.axis('equal')

        # random initial velocity
        init_velocity = scipy.stats.truncnorm.rvs((min_limit - vel_mean) / vel_sigma,
                                                  (max_limit - vel_mean) / vel_sigma,
                                                  loc=vel_mean, scale=vel_sigma, size=1)[0]
        state.ego_state.map_state.road_fstate[FS_SV] = init_vel_arr[sample_idx] = init_velocity

        # Create behavioral state
        behavioral_state = BehavioralGridState.create_from_state(state=state, logger=logger)
        action_specs = action_space.specify_goals(same_lane_recipes, behavioral_state)

        # Create arrays for T, v_0, v_T for every spec. For any None spec, take T = max_T
        T = np.array([spec.t if spec is not None else max_T for spec in action_specs])
        v_T = np.array([recipe.velocity for recipe in same_lane_recipes])
        v_0 = np.repeat(init_velocity, len(T))
        zeros = np.repeat(0, len(T))

        # create A_inv & poly_coefs_s for all specs
        A_inv = np.linalg.inv(QuarticPoly1D.time_constraints_tensor(T))
        constraints_s = np.c_[np.repeat(map_state.road_fstate[FS_SX], len(T)), v_0, zeros, v_T, zeros]
        poly_coefs_s = QuarticPoly1D.zip_solve(A_inv, constraints_s)

        # create fstates_s for all specs
        # for any spec.t < T_max, complement fstates_s to the length of T_max by adding constant velocity (v_T) states
        fstates_s = QuarticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        for i in range(fstates_s.shape[0]):
            last_t = int(T[i]/dt)
            s0 = fstates_s[i, last_t, FS_SX]
            v0 = fstates_s[i, last_t, FS_SV]
            fstates_s[i, last_t+1:, FS_SX] = s0 + np.linspace(0, v0 * (max_T - T[i]), num=len(fstates_s[i, last_t+1:, FS_SX]))
            fstates_s[i, last_t+1:, FS_SV] = v0
            fstates_s[i, last_t+1:, FS_SA] = 0
        fstates_s[..., FS_SV] = np.maximum(fstates_s[..., FS_SV], EPS)

        # create ftrajectories and convert them to ctrajectories
        ftrajectories = np.dstack((fstates_s,
                                   np.full((fstates_s.shape[0], fstates_s.shape[1]), map_state.road_fstate[FS_DX]),
                                   np.zeros((fstates_s.shape[0], fstates_s.shape[1], 2))))
        ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)

        # calculate ground truth (maximal lateral acceleration) for each ctrajectory
        max_lat_acc = np.max(np.abs(ctrajectories[..., C_K]) * ctrajectories[..., C_V] ** 2, axis=1)
        max_accel_mat[sample_idx, :] = max_lat_acc

        # encode curvature and append it to curvatures_arr
        encoded_curvature = encode_curvature(frenet.k[from_idx:from_idx+int(road_length), 0])
        curvatures_arr = np.concatenate((curvatures_arr, encoded_curvature[np.newaxis]), axis=0)

        #plt.subplot(2, 1, 2)
        #plt.plot(frenet.k[from_idx:till_idx, 0])
        #plt.show()
        #print('%.2f %.2f %s %.2f\n' % (v_0, v_T, curvatures, max_lat_acc), end=" ")

    print('good accelerations percent = %f' % (float(np.sum(max_accel_mat < 3)) / (num_samples * len(same_lane_recipes))))

    # return np.c_[vel_arr*0.1, curvatures_arr*1000], max_accel_arr, np.array([0, 1, 0, 1, 0, 1])

    sq_vel = np.square(init_vel_arr)
    curve_radius = np.reciprocal(curvatures_arr)

    vel_normalize_factor = 2. / vel_mean**2
    curv_normalize_factor = 0.2 / vel_mean**2

    return np.c_[sq_vel * vel_normalize_factor, curve_radius * curv_normalize_factor], max_accel_mat


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

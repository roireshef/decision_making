import numpy as np

from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_DX, FS_SX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from mapping.src.service.map_service import MapService


def test_getTimeByLongitude():
    """
    Calculate time t by given longitude lon, and verify that the sampled longitude at t is equal to lon.
    """
    init_fstate = np.array([0, 0, 0, 0, 0, 0])
    target_fstate = np.array([100, 0, 0, 0, 0, 0])
    T = 10
    road_id = 20

    A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

    constraints_s = np.concatenate((init_fstate[:FS_DX], target_fstate[:FS_DX]))
    constraints_d = np.concatenate((init_fstate[FS_DX:], target_fstate[FS_DX:]))
    poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
    poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]
    road_frenet = MapService.get_instance()._rhs_roads_frenet[road_id]

    samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=10, T_s=T, T_d=T, frenet_frame=road_frenet,
                                                      poly_s_coefs=poly_coefs_s, poly_d_coefs=poly_coefs_d)

    lon = 30.
    # calculate t, for which the samplable trajectory passes lon
    t = samplable_trajectory.get_time_from_longitude(road_id, lon)
    # sample the trajectory at t
    lon1 = samplable_trajectory.sample_frenet(np.array([t]))[0, FS_SX]
    assert np.isclose(lon, lon1)

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE
from decision_making.src.planning.trajectory.optimal_control.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.types import C_X, C_Y, C_YAW, C_V, C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import EgoState, ObjectSize
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import TrajectoryPlanningFacadeMock
from decision_making.test.planning.trajectory.utils import RouteFixture
from mapping.src.transformations.geometry_utils import CartesianFrame
import numpy as np

from rte.python.logger.AV_logger import AV_Logger


def test_isActualStateCloseToExpectedState_closeTranslatedOnlyEgoState_returnsTrue():
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame(route_points)
    poly_s_coefs = np.array([4.58963156e-01, -2.31312529e+00, 3.49717428e+00,
                             -9.50993030e-08, 5.99999999e+00, 9.50994838e-10])

    poly_d_coefs = np.array([5.58098455e-01, -2.05184044e+00, 1.97018719e+00,
                             3.80399214e-08, -1.14119374e-08, 5.00000000e-01])

    samplable_trajectory = SamplableWerlingTrajectory(1000, 1011.5, frenet, poly_s_coefs, poly_d_coefs)

    facade = TrajectoryPlanningFacadeMock(None, AV_Logger.get_logger(""), None, None, samplable_trajectory)

    exact_desired_state = samplable_trajectory.sample(np.array([1001]))[0]
    close_state = EgoState(-1, 1001e9, exact_desired_state[C_X] + 0.1, exact_desired_state[C_Y] + 0.1,
                           DEFAULT_OBJECT_Z_VALUE, exact_desired_state[C_YAW], ObjectSize(0, 0, 0), 1.0,
                           exact_desired_state[C_V], 0.0, exact_desired_state[C_A], 0.0, 0.0)

    assert facade._is_actual_state_close_to_expected_state(close_state)


def test_isActualStateCloseToExpectedState_nonCloseTranslatedOnlyEgoState_returnsFalse(ego_state_fix):
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame(route_points)
    poly_s_coefs = np.array([4.58963156e-01, -2.31312529e+00, 3.49717428e+00,
                             -9.50993030e-08, 5.99999999e+00, 9.50994838e-10])

    poly_d_coefs = np.array([5.58098455e-01, -2.05184044e+00, 1.97018719e+00,
                             3.80399214e-08, -1.14119374e-08, 5.00000000e-01])

    samplable_trajectory = SamplableWerlingTrajectory(1000, 1011.5, frenet, poly_s_coefs, poly_d_coefs)

    facade = TrajectoryPlanningFacadeMock(None, AV_Logger.get_logger(""), None, None, samplable_trajectory)

    exact_desired_state = samplable_trajectory.sample(np.array([1001]))[0]
    close_state = EgoState(-1, 1001e9, exact_desired_state[C_X] + 200, exact_desired_state[C_Y] + 200,
                           DEFAULT_OBJECT_Z_VALUE, exact_desired_state[C_YAW], ObjectSize(0, 0, 0), 1.0,
                           exact_desired_state[C_V], 0.0, exact_desired_state[C_A], 0.0, 0.0)

    assert not facade._is_actual_state_close_to_expected_state(close_state)
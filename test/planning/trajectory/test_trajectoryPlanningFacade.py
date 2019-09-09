from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment

from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade

from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import C_X, C_Y, C_YAW, C_V, C_A, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.behavioral.state import ObjectSize, EgoState
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import TrajectoryPlanningFacadeMock
from decision_making.test.planning.trajectory.utils import RouteFixture
from rte.python.logger.AV_logger import AV_Logger
import numpy as np


def test_isActualStateCloseToExpectedState_closeTranslatedOnlyEgoState_returnsTrue():
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame.fit(route_points)
    poly_s_coefs = np.array([4.58963156e-01, -2.31312529e+00, 3.49717428e+00,
                             -9.50993030e-08, 5.99999999e+00, 9.50994838e-10])

    poly_d_coefs = np.array([5.58098455e-01, -2.05184044e+00, 1.97018719e+00,
                             3.80399214e-08, -1.14119374e-08, 5.00000000e-01])

    samplable_trajectory = SamplableWerlingTrajectory(1000, 1011.5, 1011.5, 1011.5, frenet, poly_s_coefs, poly_d_coefs)

    facade = TrajectoryPlanningFacadeMock(None, AV_Logger.get_logger(""), None, None, samplable_trajectory)

    exact_desired_state = samplable_trajectory.sample(np.array([1001]))[0]
    close_state = EgoState.create_from_cartesian_state(-1, 1001e9, np.array([exact_desired_state[C_X] + 0.1, exact_desired_state[C_Y] + 0.1,
                                                                             exact_desired_state[C_YAW], exact_desired_state[C_V], exact_desired_state[C_A], 0.0]), ObjectSize(0, 0, 0), 1.0, False)

    assert LocalizationUtils.is_actual_state_close_to_expected_state(close_state, facade._last_trajectory,
                                                                     facade.logger,
                                                                     TrajectoryPlanningFacadeMock.__class__.__name__)


def test_isActualStateCloseToExpectedState_nonCloseTranslatedOnlyEgoState_returnsFalse():
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame.fit(route_points)
    poly_s_coefs = np.array([4.58963156e-01, -2.31312529e+00, 3.49717428e+00,
                             -9.50993030e-08, 5.99999999e+00, 9.50994838e-10])

    poly_d_coefs = np.array([5.58098455e-01, -2.05184044e+00, 1.97018719e+00,
                             3.80399214e-08, -1.14119374e-08, 5.00000000e-01])

    samplable_trajectory = SamplableWerlingTrajectory(1000, 1011.5, 1011.5, 1011.5, frenet, poly_s_coefs, poly_d_coefs)

    facade = TrajectoryPlanningFacadeMock(None, AV_Logger.get_logger(""), None, None, samplable_trajectory)

    exact_desired_state = samplable_trajectory.sample(np.array([1001]))[0]
    close_state = EgoState.create_from_cartesian_state(-1, 1001e9, np.array([exact_desired_state[C_X] + 200, exact_desired_state[C_Y] + 200,
                                                                             exact_desired_state[C_YAW],
                                                                             exact_desired_state[C_V], exact_desired_state[C_A], 0.0]), ObjectSize(0, 0, 0), 1.0, False)

    assert not LocalizationUtils.is_actual_state_close_to_expected_state(close_state, facade._last_trajectory,
                                                                         facade.logger,
                                                                         TrajectoryPlanningFacadeMock.__class__.__name__)


def test_getStateWithExpectedEgo_getsState_modifiesEgoStateInIt(state):
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)

    frenet = FrenetSerret2DFrame.fit(route_points)
    poly_s_coefs = np.array([4.58963156e-01, -2.31312529e+00, 3.49717428e+00,
                             -9.50993030e-08, 5.99999999e+00, 9.50994838e-10])

    poly_d_coefs = np.array([5.58098455e-01, -2.05184044e+00, 1.97018719e+00,
                             3.80399214e-08, -1.14119374e-08, 5.00000000e-01])

    samplable_trajectory = SamplableWerlingTrajectory(1000, 1011.5, 1011.5, 1011.5, frenet, poly_s_coefs, poly_d_coefs)

    facade = TrajectoryPlanningFacadeMock(None, AV_Logger.get_logger(""), None, None, samplable_trajectory)

    # expected ego is the ego-state sampled from the facade._last_trajectory at time given by state.ego_state.timestamp
    state.ego_state.timestamp = 1001 * 1e9
    modified_state = facade._get_state_with_expected_ego(state)

    sampled_ego_state_vec = samplable_trajectory.sample(np.array([1001]))[0]

    # assert that ego-state has been changed
    np.testing.assert_almost_equal(modified_state.ego_state.x, sampled_ego_state_vec[C_X])
    np.testing.assert_almost_equal(modified_state.ego_state.y, sampled_ego_state_vec[C_Y])
    np.testing.assert_almost_equal(modified_state.ego_state.yaw, sampled_ego_state_vec[C_YAW])
    np.testing.assert_almost_equal(modified_state.ego_state.velocity, sampled_ego_state_vec[C_V])
    np.testing.assert_almost_equal(modified_state.ego_state.acceleration, sampled_ego_state_vec[C_A])
    np.testing.assert_almost_equal(modified_state.ego_state.curvature, sampled_ego_state_vec[C_K])


def test_prepareVisualizationMsg_withObjects_returnsValidMsg(state):
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)
    frenet = FrenetSerret2DFrame.fit(route_points)
    gff = GeneralizedFrenetSerretFrame.build([frenet], [FrenetSubSegment(1, 0, frenet.s_max)])

    ctrajectories = np.array([[[1, 2, 3, 4, 5, 6]]])
    planning_horizon = 0.1

    predictor = RoadFollowingPredictor(AV_Logger.get_logger(""))
    msg = TrajectoryPlanningFacade._prepare_visualization_msg(state, ctrajectories, planning_horizon, predictor, gff)

    assert len(msg.data.as_actors_predictions) == len(state.dynamic_objects)


def test_prepareVisualizationMsg_withoutObjects_returnsValidMsg(state):
    route_points = RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5)
    frenet = FrenetSerret2DFrame.fit(route_points)
    gff = GeneralizedFrenetSerretFrame.build([frenet], [FrenetSubSegment(1, 0, frenet.s_max)])

    ctrajectories = np.array([[[1, 2, 3, 4, 5, 6]]])
    planning_horizon = 0.1

    state.dynamic_objects = []

    predictor = RoadFollowingPredictor(AV_Logger.get_logger(""))
    msg = TrajectoryPlanningFacade._prepare_visualization_msg(state, ctrajectories, planning_horizon, predictor, gff)

    assert len(msg.data.as_actors_predictions) == len(state.dynamic_objects)
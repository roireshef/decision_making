from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.geometry_utils import *
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.planning.trajectory.utils import RouteFixture


def test_werlingPlanner_toyScenario_noException():
    route_points = CartesianFrame.add_yaw_and_derivatives(
        RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5))

    v0 = 8
    vT = 8
    v_min = 0
    v_max = 10
    a_min = -5
    a_max = 5
    T = 3.5

    goal = np.concatenate((route_points[len(route_points) // 2, [R_X, R_Y, R_THETA]], [vT]))

    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=7, y=-.5, z=0, yaw=0, size=ObjectSize(1.5, 0.5, 0),
                      road_localization=None, confidence=1.0, v_x=0.0, v_y=0.0, acceleration_lon=0.0, yaw_deriv=0.0),
        DynamicObject(obj_id=0, timestamp=0, x=13, y=2, z=0, yaw=np.pi / 4, size=ObjectSize(1.5, 0.5, 0),
                      road_localization=None, confidence=1.0, v_x=0.0, v_y=0.0, acceleration_lon=0.0, yaw_deriv=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=None, road_localization=None, confidence=1.0,
                   v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, yaw_deriv=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    cost_params = TrajectoryCostParams(time=T, ref_deviation_weight=10.0, lane_deviation_weight=10.0,
                                       obstacle_weight=20000.0, left_lane_offset=.5, right_lane_offset=.5,
                                       left_deviation_exp=100.0, right_deviation_exp=100.0, obstacle_offset=.2,
                                       obstacle_exp=5.0, v_x_min_limit=v_min, v_x_max_limit=v_max, a_x_min_limit=a_min,
                                       a_x_max_limit=a_max)

    planner = WerlingPlanner(None)
    _, _, debug = planner.plan(state, route_points[:, :2], goal, cost_params)

    assert True

    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # p1 = fig.add_subplot(211)
    # p2 = fig.add_subplot(212)
    # plottable_obs = [PlottableSigmoidStatic2DBoxObstacle.from_object_state(o, cost_params.obstacle_exp, cost_params.obstacle_offset)
    #                  for o in state.static_objects]
    # WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    # WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    # WerlingVisualizer.plot_route(p1, debug['ref_route'])
    # WerlingVisualizer.plot_route(p2, debug['ref_route'])
    #
    # WerlingVisualizer.plot_best(p2, debug['trajectories'][0])
    # WerlingVisualizer.plot_alternatives(p1, debug['trajectories'])
    #
    # print(debug['costs'])
    #
    # # WerlingVisualizer.plot_route(p1, route_points)
    #
    # fig.show()
    # fig.clear()

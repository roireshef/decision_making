from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.columns import R_X, R_Y, R_THETA
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.planning.trajectory.utils import *
from mapping.src.transformations.geometry_utils import *
import time

def test_werlingPlanner_toyScenario_noException():
    route_points = CartesianFrame.add_yaw_and_derivatives(
        RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5))

    v0 = 6
    vT = 10
    v_min = 0
    v_max = 10
    a_min = -5
    a_max = 5
    T = 3.5

    goal = np.concatenate((route_points[len(route_points) // 2, [R_X, R_Y, R_THETA]], [vT]))

    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=7, y=-.5, z=0, yaw=0, size=ObjectSize(1.5, 0.5, 0),
                      road_localization=None, confidence=1.0, v_x=0.0, v_y=0.0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=0, x=13, y=2, z=0, yaw=np.pi / 4, size=ObjectSize(1.5, 0.5, 0),
                      road_localization=None, confidence=1.0, v_x=0.0, v_y=0.0, acceleration_lon=0.0, omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=None, road_localization=None, confidence=1.0,
                   v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    cost_params = TrajectoryCostParams(left_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       right_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       left_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       right_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       left_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       right_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       obstacle_cost=SigmoidFunctionParams(100, 10.0, 0.3),
                                       dist_from_ref_sq_cost_coef=1.0,
                                       velocity_limits=np.array([v_min, v_max]),
                                       acceleration_limits=np.array([a_min, a_max]))

    planner = WerlingPlanner(None)

    start_time = time.time()

    _, _, samplable, debug = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                               time=T, cost_params=cost_params)

    end_time = time.time() - start_time

    assert True

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)
    plottable_obs = [PlottableSigmoidStatic2DBoxObstacle.from_object(o, cost_params.obstacle_cost.k, cost_params.obstacle_cost.offset)
                     for o in state.dynamic_objects]
    WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    WerlingVisualizer.plot_route(p1, debug.reference_route)
    WerlingVisualizer.plot_route(p2, debug.reference_route)

    WerlingVisualizer.plot_best(p2, debug.trajectories[0])
    WerlingVisualizer.plot_alternatives(p1, debug.trajectories)

    print(debug.costs)

    # WerlingVisualizer.plot_route(p1, route_points)

    fig.show()
    fig.clear()

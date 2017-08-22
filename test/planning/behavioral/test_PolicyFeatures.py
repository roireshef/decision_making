import numpy as np
from decision_making.src.map.map_api import MapAPI
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.default_policy import DefaultPolicyConfig
from decision_making.src.planning.behavioral.policy_features import DefaultPolicyFeatures
from decision_making.src.state.state import EgoState, RoadLocalization, RelativeRoadLocalization, DynamicObject, \
    ObjectSize
from rte.python.logger.AV_logger import AV_Logger


class MapMock(MapAPI):
    def __init__(self):
        pass


def test_get_closest_object_on_lane_ComplexScenraio_success():
    policy_config = DefaultPolicyConfig()
    map_api = MapMock()
    logger = AV_Logger.get_logger("Policy features test")
    navigation_plan = NavigationPlanMsg(road_ids=[0])

    # ego state at (0,0,0)
    road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=0.0, intra_lane_lat=0.0, road_lon=0.0,
                                         intra_lane_yaw=0.0)
    relative_road_localization = RelativeRoadLocalization(rel_lat=0.0, rel_lon=0.0, rel_yaw=0.0)
    ego_state = EgoState(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, yaw_deriv=0.0, map_api=map_api, road_localization=road_localization,
                         rel_road_localization=relative_road_localization)

    # obstacle at (10 + i*1.0, 1.5 + i*0.5, 0)
    objects_list = list()
    for i in range(10):
        road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=1.5 + i * 0.5, intra_lane_lat=0.0,
                                             road_lon=10.0 + i * 1.0,
                                             intra_lane_yaw=0.0)
        relative_road_localization = RelativeRoadLocalization(rel_lat=1.5 + i * 0.5, rel_lon=10.0 + i * 1.0,
                                                              rel_yaw=0.0)
        static_object = DynamicObject(obj_id=1, timestamp=0, x=10.0 + i * 1.0, y=1.5 + i * 0.5, z=0.0, yaw=0.0,
                                      size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                      confidence=1.0, v_x=0.0, v_y=0.0,
                                      acceleration_lon=0.0, yaw_deriv=0.0, ego_state=ego_state, map_api=map_api,
                                      road_localization=road_localization,
                                      rel_road_localization=relative_road_localization)

        objects_list.append(static_object)

    behavioral_state = BehavioralState(logger=logger, map_api=map_api, navigation_plan=navigation_plan,
                                       ego_state=ego_state, dynamic_objects=objects_list)

    lat_options_in_meters = np.array(range(20)) * 0.5 - 0.5
    closest_objects_per_lat_options = DefaultPolicyFeatures.get_closest_object_on_lane(policy_config=policy_config,
                                                                                       behavioral_state=behavioral_state,
                                                                                       lat_options=lat_options_in_meters)

    assert np.all(np.isinf(closest_objects_per_lat_options[-1:-4:-1]))
    assert np.isinf(closest_objects_per_lat_options[0])
    assert np.all(np.abs(closest_objects_per_lat_options[1:-3].reshape([-1]) - np.array(
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])) < 0.01)

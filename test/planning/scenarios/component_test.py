from typing import List

import numpy as np
import math


from decision_making.src.state.state import OccupancyState, RoadLocalization, EgoState, ObjectSize, DynamicObject, State

import decision_making.test.planning.scenarios.run_dm_processes as run_dm
from mapping.src.service.map_service import MapService




def component_test():
    #This function defines a list of states, runs the DM processes with the original facades,
    # except for a mock version of the state module facade which receives the state list to transmit over DDS


    stateList : List[State] = list()

    #The default, currently is the demo scenario
    demo_map_api=MapService.get_instance()
    road_id=20
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))


    lane_width = demo_map_api.get_road(road_id).lane_width



    #State 1
    ego_road_lon=5.0
    ego_road_lat=lane_width*0.5 #to place ego in middle of right lane
    ego_lon_speed=25.0*1000/3600
    ego_front=ego_road_lon+2.5/2


    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id,ego_road_lon,ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)


    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1], z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)


    # Generate objects with at the following locations:

    dynamic_objects : List[DynamicObject] = list()

    #Dynamic object 1
    obj_id=0
    obj_road_lon=5.0+20.0
    obj_road_lat=lane_width*0.5
    obj_speed=15*1000/3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                           obj_road_lat)


    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x=obj_speed*math.cos(obj_yaw)
    obj_v_y=obj_speed*math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0], y=obj_global_localization[1],
                                       z=obj_global_localization[2], yaw=obj_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                       road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    #Dynamic object 2
    obj_id=1
    obj_road_lon=5.0+80.
    obj_road_lat=lane_width*1.5
    obj_speed=25*1000/3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                           obj_road_lat)


    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x=obj_speed*math.cos(obj_yaw)
    obj_v_y=obj_speed*math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0], y=obj_global_localization[1],
                                       z=obj_global_localization[2], yaw=obj_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                       road_localization=obj_road_localization)


    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    #State 2
    ego_road_lon=5.0
    ego_road_lat=lane_width*0.5 #to place ego in middle of right lane
    ego_lon_speed=25.0*1000/3600
    ego_front=ego_road_lon+2.5/2


    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id,ego_road_lon,ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)


    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1], z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)


    # Generate objects with at the following locations:

    dynamic_objects : List[DynamicObject] = list()

    #Dynamic object 1
    obj_id=0
    obj_road_lon=5.0+15.0
    obj_road_lat=lane_width*0.5
    obj_speed=15*1000/3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                           obj_road_lat)


    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x=obj_speed*math.cos(obj_yaw)
    obj_v_y=obj_speed*math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0], y=obj_global_localization[1],
                                       z=obj_global_localization[2], yaw=obj_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                       road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    #Dynamic object 2
    obj_id=1
    obj_road_lon=5.0+80.
    obj_road_lat=lane_width*1.5
    obj_speed=25*1000/3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                           obj_road_lat)


    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x=obj_speed*math.cos(obj_yaw)
    obj_v_y=obj_speed*math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0], y=obj_global_localization[1],
                                       z=obj_global_localization[2], yaw=obj_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                       confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                       road_localization=obj_road_localization)


    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)


    # State 4
    ego_road_lon = 10.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    # State 5
    ego_road_lon = 15.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    # State 6
    ego_road_lon = 20.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    # State 7
    ego_road_lon = 25.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    # State 8
    ego_road_lon = 30.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)

    # State 9
    ego_road_lon = 40.0
    ego_road_lat = lane_width * 1.5  # to place ego in middle of right lane
    ego_lon_speed = 25.0 * 1000 / 3600
    ego_front = ego_road_lon + 2.5 / 2

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    ego_global_localization, ego_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, ego_road_lon,
                                                                                       ego_road_lat)

    ego_road_localization = RoadLocalization(road_id, int(lane), ego_road_lat, intra_lane_lat, ego_road_lon, ego_yaw)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_global_localization[0], y=ego_global_localization[1],
                         z=ego_global_localization[2],
                         yaw=ego_yaw, size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=ego_lon_speed, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=ego_road_localization)

    # Generate objects with at the following locations:

    dynamic_objects: List[DynamicObject] = list()

    # Dynamic object 1
    obj_id = 0
    obj_road_lon = 5.0 + 15.0
    obj_road_lat = lane_width * 0.5
    obj_speed = 15 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    # Dynamic object 2
    obj_id = 1
    obj_road_lon = 5.0 + 80.
    obj_road_lat = lane_width * 1.5
    obj_speed = 35 * 1000 / 3600

    lane = np.math.floor(ego_road_lat / lane_width)
    intra_lane_lat = ego_road_lat - lane * lane_width

    obj_global_localization, obj_yaw = demo_map_api.convert_road_to_global_coordinates(road_id, obj_road_lon,
                                                                                       obj_road_lat)

    obj_road_localization = RoadLocalization(road_id, int(lane), obj_road_lat, intra_lane_lat, obj_road_lon,
                                             obj_yaw)

    obj_v_x = obj_speed * math.cos(obj_yaw)
    obj_v_y = obj_speed * math.sin(obj_yaw)

    dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_global_localization[0],
                                   y=obj_global_localization[1],
                                   z=obj_global_localization[2], yaw=obj_yaw,
                                   size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                   confidence=1.0, v_x=obj_v_x, v_y=obj_v_y, acceleration_lon=0.0, omega_yaw=0.0,
                                   road_localization=obj_road_localization)

    dynamic_objects.append(dynamic_object)

    test_state = State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)

    stateList.append(test_state)



    run_dm.run_processes(stateList)


component_test()


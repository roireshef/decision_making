import time

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.state import *
from rte.python.logger.AV_logger import AV_Logger


class StateModule(DmModule):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger):
        super().__init__(dds, logger)
        self.state = State.create_empty()

    def _start_impl(self):
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self.__dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self.__self_localization_callback)
        self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self.__occupancy_state_callback)

    def _stop_impl(self):
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def _periodic_action_impl(self):
        state = State.create_empty()
        state.__class__ = State
        state.from_state(self.state)

        # # Publish dummy state
        # road_localization = RoadLocalization(0, 0, 0, 0, 0, 0, 0)
        # lanes_structure = LanesStructure(np.array([]), np.array([]))
        # occupancy_state = OccupancyState(0, np.array([0.0]), np.array([0.0]))
        # dynamic_objects = [DynamicObject(0, 0, 0, 0, 0, 0, ObjectSize(0, 0, 0),
        #                                          road_localization, 0, 0, 0, 0, 0, 0)]
        # ego_state = EgoState(0, 0, 0, 0, 0, 0, ObjectSize(0, 0, 0),
        #                              road_localization, 0, 0, 0, 0, 0, 0, 0)
        # perceived_road = PerceivedRoad(0, [lanes_structure], 0)
        #
        # state = State(occupancy_state=occupancy_state,
        #                                dynamic_objects=dynamic_objects, ego_state=ego_state,
        #                                perceived_road=perceived_road)

        state_serialized = state.serialize()
        self.logger.info(state_serialized)

    def __dynamic_obj_callback(self, objects: dict):
        self.logger.info("got dynamic objects %s", objects)

        timestamp = objects["timestamp"]
        dyn_obj_list_dict = objects["timestamp"]["dynamic_objects"]

        dyn_obj_list = self.state.dynamic_objects
        for dyn_obj_dict in dyn_obj_list_dict:
            id = dyn_obj_dict["id"]
            x = dyn_obj_dict["location"]["x"]
            y = dyn_obj_dict["location"]["y"]
            z = 0
            yaw = dyn_obj_dict["bbox"]["yaw"]
            confidence = 1
            localization_confidence = dyn_obj_dict["location"]["confidence"]
            length = dyn_obj_dict["bbox"]["length"]
            width = dyn_obj_dict["bbox"]["width"]
            height = dyn_obj_dict["bbox"]["height"]
            size = ObjectSize(length, width, height)
            road_localization = self.__get_road_lane_localization(dyn_obj_dict["road_localization"],
                                                                  dyn_obj_dict["lane_localization"])
            v_x = dyn_obj_dict["velocity"]["v_x"]
            v_y = dyn_obj_dict["velocity"]["v_y"]

            dyn_obj = DynamicObject(id, timestamp, x,y,z, yaw, size, road_localization,
                                    confidence, localization_confidence, v_x, v_y)
            dyn_obj_list.append(dyn_obj)

    def __self_localization_callback(self, ego_localization: dict):
        self.logger.debug("got self localization %s", ego_localization)

        confidence = 1
        timestamp = ego_localization["timestamp"]
        x = ego_localization["location"]["x"]
        y = ego_localization["location"]["y"]
        z = 0
        localization_confidence = ego_localization["location"]["confidence"]
        yaw = ego_localization["yaw"]
        v_x = ego_localization["velocity"]["v_x"]
        v_y = ego_localization["velocity"]["v_y"]
        size = ObjectSize(0,0,0)
        road_localization = self.__get_road_lane_localization(ego_localization["road_localization"],
                                                              ego_localization["lane_localization"])
        self.state.ego_state = EgoState(0, timestamp, x,y,z, yaw, size, road_localization,
                                        confidence, localization_confidence, v_x, v_y, 0)

    def __occupancy_state_callback(self, occupancy: dict):
        self.logger.debug("got occupancy status %s", occupancy)
        timestamp = occupancy["timestamp"]
        points_list_dict = occupancy["points_list"]
        points_list = []
        confidence_list = []
        for pnt_dict in points_list_dict:
            pnt = [pnt_dict["x"], pnt_dict["y"], pnt_dict["z"]]
            points_list.append(pnt)
            confidence_list.append(pnt_dict["confidence"])
        self.state.occupancy_state = OccupancyState(timestamp, points_list, confidence_list)

    def __actuator_status_callback(self, actuator: dict):
        self.logger.debug("got actuator status %s", actuator)
        self.state.ego_state.steering_angle = actuator["steering_angle"]
    """
    def __percieved_road_callback(self, percieved_road: dict):
        self.logger.info("got percieved_road status %s", percieved_road)
        timestamp = percieved_road["timestamp"]
        num_of_lanes = percieved_road["num_of_lanes"]
        lane_structures = percieved_road["lane_structures"]
        
        self_lane_localization = percieved_road["self_lane_localization"]
        confidence = percieved_road["confidence"]
        self.state.perceived_road = PerceivedRoad(timestamp, lane_structures, confidence)
    """
    def __get_road_lane_localization(self, road_loc: dict, lane_loc: dict):
        road_id = road_loc["road_id"]
        road_lon = road_loc["road_longditude"]
        road_confidence = road_loc["confidence"]
        lane = lane_loc["lane_num"]
        intra_lane_lat = lane_loc["intra_lane_latitude"]
        intra_lane_yaw = lane_loc["intra_lane_yaw"]
        lane_confidence = lane_loc["confidence"]
        road_localization = RoadLocalization(road_id, lane, intra_lane_lat, road_lon, intra_lane_yaw, road_confidence,
                                                 lane_confidence)
        return road_localization


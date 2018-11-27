from decision_making.src.messages.scene_static_message import SceneStatic, SceneLaneSegment, SceneRoadSegment


class SceneModel:
    """
    Data layer. Holds the data from SceneStatic. Currently only holds the message. Might be a <<Singleton>>
    """
    __instance = None

    @classmethod
    def get_instance(cls) -> 'SceneModel':
        if cls.__instance is None:
            cls.__instance = SceneModel()
        return cls.__instance

    def add_scene_static(self, message: SceneStatic) -> None:
        self.messages.append(message)

    def get_scene_static(self) -> SceneStatic:
        return self._messages[-1]

    def get_lane(self, lane_id: int) -> SceneLaneSegment:
        scene_static = self.get_scene_static()
        lanes = [lane for lane in scene_static.s_Data.as_scene_lane_segment if
                 lane.e_i_lane_segment_id == lane_id]
        assert len(lanes) == 1
        return lanes[0]

    def get_road_segment(self, road_id: int) -> SceneRoadSegment:
        scene_static = self.get_scene_static()
        road_segments = [road_segment for road_segment in scene_static.s_Data.as_scene_road_segment if
                         road_segment.e_Cnt_road_id == road_id]
        assert len(road_segments) == 1
        return road_segments[0]

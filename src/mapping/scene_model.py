from decision_making.src.messages.scene_static_message import SceneStatic, SceneLaneSegment, SceneRoadSegment


class SceneModel:
    """
    Data layer. Holds the data from SceneStatic (currently expecting a single message).
     A <<Singleton>>
    """
    __instance = None

    def __init__(self) -> None:
        self._messages = []



    @classmethod
    def get_instance(cls) -> None:
        """
        :return: The instance of SceneModel
        """
        if cls.__instance is None:
            cls.__instance = SceneModel()
        return cls.__instance

    def add_scene_static(self, message: SceneStatic) -> None:
        """
        Add a SceneStatic message to the model. Currently this assumes there is only
        a single message
        :param message:  The SceneStatic message
        :return:
        """
        self._messages.append(message)

    def get_scene_static(self) -> SceneStatic:
        """
        Gets the last message in list
        :return:  The SceneStatic message
        """
        if len(self._messages) == 0:
            raise ValueError('Scene model is empty')
        return self._messages[-1]

    def get_lane(self, lane_id: int) -> SceneLaneSegment:
        """
        Retrieves lane by lane_id  according to the last message
        :param lane_id:
        :return:
        """
        scene_static = self.get_scene_static()
        lanes = [lane for lane in scene_static.s_Data.as_scene_lane_segment if
                 lane.e_i_lane_segment_id == lane_id]
        assert len(lanes) == 1
        return lanes[0]

    def get_road_segment(self, road_id: int) -> SceneRoadSegment:
        """

        Retrieves road by road_id  according to the last message
        :param road_id:
        :return:
        """
        scene_static = self.get_scene_static()
        road_segments = [road_segment for road_segment in scene_static.s_Data.as_scene_road_segment if
                         road_segment.e_Cnt_road_segment_id == road_id]
        assert len(road_segments) == 1
        return road_segments[0]


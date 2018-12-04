from decision_making.src.messages.scene_static_message import SceneStatic, SceneLaneSegment, SceneRoadSegment


class SceneModel:
    """
    Data layer. Holds the data from SceneStatic (currently expecting a single message).
     A <<Singleton>>
    """
    __instance = None

    def __init__(self) -> None:
        self._message = None



    @classmethod
    def get_instance(cls) -> None:
        """
        :return: The instance of SceneModel
        """
        if cls.__instance is None:
            cls.__instance = SceneModel()
        return cls.__instance

    def set_scene_static(self, message: SceneStatic) -> None:
        """
        Add a SceneStatic message to the model. Currently this assumes there is only
        a single message
        :param message:  The SceneStatic message
        :return:
        """
        self._message = message

    def get_scene_static(self) -> SceneStatic:
        """
        Gets the last message in list
        :return:  The SceneStatic message
        """
        if self._message is None:
            raise ValueError('Scene model is empty')
        return self._message



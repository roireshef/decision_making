from decision_making.src.exceptions import SceneModelIsEmpty
from decision_making.src.messages.scene_static_message import SceneStatic


class SceneStaticModel:
    """
    Data layer. Holds the data from SceneStatic (currently expecting a single message).
     A <<Singleton>>
    """
    __instance = None

    def __init__(self) -> None:
        self._scene_static_message = None

    @classmethod
    def get_instance(cls) -> 'SceneStaticModel':
        """
        :return: The instance of SceneModel
        """
        if cls.__instance is None:
            cls.__instance = SceneStaticModel()
        return cls.__instance

    def set_scene_static(self, scene_static_message: SceneStatic) -> None:
        """
        Add a SceneStatic message to the model. Currently this assumes there is only
        a single message
        :param scene_static_message:  The SceneStatic message
        :return:
        """
        self._scene_static_message = scene_static_message

    def get_scene_static(self) -> SceneStatic:
        """
        Gets the last message in list
        :return:  The SceneStatic message
        """
        if self._scene_static_message is None:
            raise SceneModelIsEmpty('Scene static model is empty')
        return self._scene_static_message



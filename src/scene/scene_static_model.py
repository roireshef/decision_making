
class SceneStaticModel:
    """
    Data layer. Holds the data from SceneStatic (currently expecting a single message).
     A <<Singleton>>
    """
    __instance = None

    def __init__(self) -> None:
        self._scene_static_message = None

    @classmethod
    def get_instance(cls) -> None:
        """
        :return: The instance of SceneStaticModel
        """
        if cls.__instance is None:
            cls.__instance = SceneStaticModel()
        return cls.__instance

    @property
    def get_scene_static(self):
        return self._scene_static_message

    def set_scene_static(self, message):
        self._scene_static_message = message

import os.path


class Paths:
    @staticmethod
    def get_repo_path():
        # type: () -> str
        return os.path.dirname(__file__)

    @staticmethod
    def get_lib_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_lib_path(), filename)

    @staticmethod
    def get_lib_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'lib')

    @staticmethod
    def get_config_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_config_path(), filename)

    @staticmethod
    def get_config_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'config')

    @staticmethod
    def get_resource_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_resource_path(), filename)

    @staticmethod
    def get_resource_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'resources')

    @staticmethod
    def get_scene_static_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'resources/scene_static_mocks')

    @staticmethod
    def get_scene_dynamic_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'resources/scene_dynamic_mocks')

    @staticmethod
    def get_scene_static_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_scene_static_path(), filename)

    @staticmethod
    def get_scene_dynamic_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_scene_dynamic_path(), filename)

    @staticmethod
    def get_resources_path() -> str:
        return os.path.join(os.path.dirname(__file__), 'resources')

    @staticmethod
    def get_checkpoints_dir_path() -> str:
        return os.path.join(Paths.get_resources_path(), 'checkpoints')

    @staticmethod
    def get_repo_absolute_dir_path(relative_dir: str) -> str:
        return os.path.join(Paths.get_repo_path(), relative_dir)

    @staticmethod
    def get_checkpoint(name: str) -> str:
        return os.path.join(Paths.get_checkpoints_dir_path(), name)

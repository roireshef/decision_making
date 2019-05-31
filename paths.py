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
    def get_maps_path():
        # type: () -> str
        return os.path.join(os.path.dirname(__file__), 'resources/scene_static_mocks')

    @staticmethod
    def get_map_absolute_path_filename(filename):
        # type: (str) -> str
        return os.path.join(Paths.get_maps_path(), filename)

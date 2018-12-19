import os.path


class Paths:
    """
    Holds all the paths relative to the Maps project
    """

    @staticmethod
    def get_project_path():
        return os.path.join(os.path.dirname(__file__), '../')

    @staticmethod
    def get_resource_absolute_path_filename(resource, version, filename):
        # type: (str) -> str
        return os.path.join(Paths.get_resource_path(resource), version, filename)

    @staticmethod
    def get_resource_path(resource):
        # type: () -> str
        return os.path.join(Paths.get_project_path(), 'resources', resource)
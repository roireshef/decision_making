from decision_making.src.map.map_api import MapAPI

import pickle


class NaiveCacheMap(MapAPI):
    def __init__(self, map_model_filename):
        # type: (str) -> None
        """
        :param map_model_filename: python "pickle" file containing the dictionaries of map_model
        """
        super().__init__(pickle.load(open(map_model_filename, "rb")))

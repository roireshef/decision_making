from decision_making.src.map.map_api import MapAPI
import pickle


class NaiveCacheMap(MapAPI):
    def __init__(self, map_model_filename):
        """
        :param map_model_filename: python "pickle" file containing the dictionaries of map_model
        """
        super().__init__(self._load_map(map_model_filename))

    def _load_map(self, filename):
        """
        :param filename: python "pickle" file containing the dictionaries of map_model
        :return:
        """
        self.cached_map_model = pickle.load(open(filename, "rb"))

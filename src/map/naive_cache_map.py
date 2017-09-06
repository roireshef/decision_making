from decision_making.src.map.map_api import MapAPI
from decision_making.src.map.map_model import MapModel
from logging import Logger
import pickle


class NaiveCacheMap(MapAPI):
    def __init__(self, map_model_filename, logger):
        # type: (str, Logger) -> None
        """
        :param map_model: MapModel containing all relevant map data. To load .bin files (pickle), use:
        import pickle
        pickle.load(open(map_model_filename, "rb"))
        """
        map_model = pickle.load(open(map_model_filename, "rb"))
        super().__init__(map_model, logger)

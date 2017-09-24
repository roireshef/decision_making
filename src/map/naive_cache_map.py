from decision_making.src.map.constants import ROADS_MAP_TILE_SIZE
from decision_making.src.map.map_api import MapAPI
from decision_making.src.map.map_model import MapModel
from logging import Logger
import pickle


class NaiveCacheMap(MapAPI):
    def __init__(self, map_model, logger):
        # type: (str, Logger) -> None
        """
        :param map_model: MapModel containing all relevant map data.
        """
        super().__init__(map_model, logger)

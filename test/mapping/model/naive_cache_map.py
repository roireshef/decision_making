from logging import Logger

from decision_making.test.mapping.model.map_api import MapAPI
from decision_making.test.mapping.model.map_model import MapModel

class NaiveCacheMap(MapAPI):
    def __init__(self, map_model, logger):
        # type: (MapModel, Logger) -> None
        """
        :param map_model: MapModel containing all relevant map data.
        """
        super(NaiveCacheMap, self).__init__(map_model, logger)

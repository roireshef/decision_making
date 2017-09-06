from decision_making.src.map.constants import ROADS_MAP_TILE_SIZE
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

        # TODO: A PATCH TOWARDS DEMO, FIX IT
        map_model_from_file = pickle.load(open(map_model_filename, "rb"))
        map_model = MapModel(map_model_from_file.roads_data, map_model_from_file.incoming_roads,
                             map_model_from_file.outgoing_roads, map_model_from_file.xy2road_map,
                             ROADS_MAP_TILE_SIZE)
        super().__init__(map_model, logger)

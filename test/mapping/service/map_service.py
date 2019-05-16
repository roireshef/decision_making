import os
import pickle

import common_data as COMMON_DATA_ROOT
import rte.AV_config.src.Configurator as AV_Configurator
from decision_making.test.mapping.model.map_api import MapAPI
from decision_making.test.mapping.model.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger
from mapping.src.global_constants import MAP_MODULE_NAME_FOR_LOGGING


class MapService(object):
    """
    The map service
    """

    __instance = None

    @staticmethod
    def initialize(map_file=None):
        # type: (Optional[str]) -> None
        MapService.__instance = MapService.__create_instance(map_file)

    @staticmethod
    def get_instance():
        # type: () -> MapAPI
        if not MapService.__instance:
            MapService.initialize()
            # raise MapServiceNotInitialized('Please initialize the MapService')

        return MapService.__instance

    @staticmethod
    def __create_instance(map_file=None):
        # type: (Optional[str]) -> MapAPI
        logger = AV_Logger.get_logger(MAP_MODULE_NAME_FOR_LOGGING)
        try:
            if map_file == None:
                map_model = pickle.load(open(AV_Configurator.get_map_file(), "rb"))
            else:
                map_path = os.path.join(os.path.dirname(COMMON_DATA_ROOT.__file__), 'maps')
                full_map_file = os.path.join(map_path, map_file)
                map_model = pickle.load(open(full_map_file, "rb"))
            map_api = NaiveCacheMap(map_model, logger)
            return map_api
        except Exception as err:
            logger.exception("Failed to load map model from file {}: {}".format(map_file, err))

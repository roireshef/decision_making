from decision_making.src.map.map_api import MapAPI
import pickle

class NaiveCacheMap(MapAPI):
    def load_map(self, map_model_pickle_filename):
        self.cached_map_model = pickle.load(open(map_model_pickle_filename, "rb"))

from decision_making.src.map.cache_map import CacheMap
import pickle

class NaiveCacheMap(CacheMap):
    def load_map(self, map_model_pickle_filename):
        self.cached_map_model = pickle.load(open(map_model_pickle_filename, "rb"))

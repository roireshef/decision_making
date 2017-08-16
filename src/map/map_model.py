import copy


class MapModel:
    def __init__(self, roads_data={}, incoming_roads={}, outgoing_roads={}, xy2road_map={}):
        self.roads_data = copy.deepcopy(roads_data)  # dictionary: roads id -> road's attributes
        self.incoming_roads = copy.deepcopy(incoming_roads)  # dictionary: node id -> incoming roads
        self.outgoing_roads = copy.deepcopy(outgoing_roads)  # dictionary: node id -> outgoing roads
        self.xy2road_map = copy.deepcopy(xy2road_map)  # maps world coordinates to road_ids

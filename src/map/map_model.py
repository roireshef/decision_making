
class MapModel:
    def __init__(self):
        self.roads_data = {}  # dictionary: roads id -> road's attributes
        self.incoming_roads = {}  # dictionary: node id -> incoming roads
        self.outgoing_roads = {}  # dictionary: node id -> outgoing roads
        self.xy2road_map = {}  # maps world coordinates to road_ids

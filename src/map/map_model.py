
class MapModel:
    def __init__(self):
        self.roads_data = {}
        self.incoming_roads = {}
        self.outgoing_roads = {}
        self.xy2road_map = {}  # maps world coordinates to road_ids
        self.xy2road_map_cell_size = 10.0  # in meters

import copy
from typing import List, Dict
from decision_making.src.map.map_api import RoadDetails


class MapModel:
    def __init__(self, roads_data={}, incoming_roads={}, outgoing_roads={}, xy2road_map={}):
        # type: (Dict[int, RoadDetails], Dict[int, List[int]], Dict[int, List[int]], Dict[(float, float), List[int]]) -> None
        self.roads_data = copy.deepcopy(roads_data)  # dictionary: road_id -> RoadDetails
        self.incoming_roads = copy.deepcopy(incoming_roads)  # dictionary: node id -> incoming roads
        self.outgoing_roads = copy.deepcopy(outgoing_roads)  # dictionary: node id -> outgoing roads
        self.xy2road_map = copy.deepcopy(xy2road_map)  # maps world coordinates to road_ids

from typing import List


class NavigationGoal:
    def __init__(self, road_id: int, lon: float, lanes: List[int]):
        """
        Holds parameters of a navigation goal: road id, longitude, list of lanes.
        Used by value function approximation.
        :param road_id: road id from the map
        :param lon: [m] longitude of the goal relatively to the road's beginning
        :param lanes: list of lane indices of the goal
        """
        self.road_id = road_id
        self.lon = lon
        self.lanes_list = lanes

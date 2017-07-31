import numpy as np

class NavigationPlan:
    """
    This class hold the navigation plan.
    """
    def __init__(self, planned_roads_list=None):
        self.planned_roads_list = np.asarray(planned_roads_list)
        self.current_road_index = 0

    def get_current_road_id(self):
        return self.planned_roads_list[self.current_road_index]

    def get_road_index_in_plan(self, road_id):
        search_result = np.where(self.planned_roads_list == road_id)
        if len(search_result[0]) == 0:  # target road_id not found in the navigation plan
            return None
        else:
            return search_result[0][0]

    # dir=1 look forward, dir=-1 look backward
    def get_next_road(self, direction=1, road_index=None):
        if road_index is None:
            road_index = self.current_road_index

        if 0 <= road_index + direction < len(self.planned_roads_list):
            next_road_id = self.planned_roads_list[road_index+direction]
            next_road_index = road_index+direction
            return next_road_id, next_road_index
        else:
            return None, None

    def set_current_index_in_route(self, index):
        self.current_road_index = index

    def increment_current_index_in_route(self):
        self.current_road_index += 1

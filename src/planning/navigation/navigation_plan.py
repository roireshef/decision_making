import numpy as np

class NavigationPlan:
    """
    This class hold the navigation plan.
    """
    def __init__(self, road_ids: np.array):
        """
        Initialization of the navigation plan. This is an initial implementation which contains only a list o road ids.
        :param road_ids: list of road ids corresponding to the map.
        """
        self._road_ids = road_ids
        self._current_plan_index = 0

    def get_current_road_id(self):
        return self._road_ids[self._current_plan_index]

    def get_road_index_in_plan(self, road_id):
        search_result = np.where(self._road_ids == road_id)
        if len(search_result[0]) == 0:  # target road_id not found in the navigation plan
            return None
        else:
            return search_result[0][0]

    # dir=1 look forward, dir=-1 look backward
    def get_next_road(self, direction=1, road_index=None):
        if road_index is None:
            road_index = self._current_plan_index

        if 0 <= road_index + direction < len(self._road_ids):
            next_road_id = self._road_ids[road_index+direction]
            next_road_index = road_index+direction
            return next_road_id, next_road_index
        else:
            return None, None

    def set_current_index_in_route(self, index):
        self._current_plan_index = index

    def increment_current_index_in_route(self):
        self._current_plan_index += 1
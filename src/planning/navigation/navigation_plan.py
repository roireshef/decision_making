from typing import Tuple
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
        self._road_ids = np.array(road_ids)
        self._current_plan_index = 0

    def get_current_road_id(self) -> int:
        """
        get current road_id according to the current plan index
        :return: road_id
        """
        return self._road_ids[self._current_plan_index]

    def get_road_index_in_plan(self, road_id: int) -> int:
        """
        find road_id in the navigation plan and return its index
        :param road_id:
        :return: index of road_id in the plan
        """
        search_result = np.where(self._road_ids == road_id)
        if len(search_result[0]) == 0:  # target road_id not found in the navigation plan
            return type(None)
        else:
            return search_result[0][0]

    # dir=1 look forward, dir=-1 look backward
    def get_next_road(self, direction: int =1, plan_index: int =None) -> Tuple:
        """
        get next road_id according to the navigation plan
        :param direction: 1 forward (default), -1 backward
        :param road_index: optional plan index. if None, take the current_plan_index
        :return:
        """
        if plan_index is None:
            plan_index = self._current_plan_index

        if 0 <= plan_index + direction < len(self._road_ids):
            next_road_id = self._road_ids[plan_index+direction]
            next_plan_index = plan_index+direction
            return next_road_id, next_plan_index
        else:
            return None, None

    def set_current_plan_index(self, index: int):
        """
        set a new value for the current plan index
        :param index: new index
        :return:
        """
        self._current_plan_index = index

    def increment_current_plan_index(self):
        """
        increment current plan index by 1
        :return:
        """
        self._current_plan_index += 1

from typing import Union

import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class NavigationPlanMsg(DDSTypedMsg):
    """
        This class hold the navigation plan.
        It also implements function (required by MapAPI) that iterate over the roads list in the navigation plan.
        Important assumption: we assume that road_ids is a UNIQUE list, containing each value only once.
    """

    def __init__(self, road_ids: np.array):
        """
        Initialization of the navigation plan. This is an initial implementation which contains only a list o road ids.
        :param road_ids: list of road ids corresponding to the map.
        """
        self.road_ids = road_ids

    def get_road_index_in_plan(self, road_id: int) -> Union[None, int]:
        """
        Given a road_id, returns the index of this road_id in the navigation plan
        :param road_id:
        :return: index of road_id in the plan
        """
        search_result = np.where(self._road_ids == road_id)
        if len(search_result[0]) == 0:  # target road_id not found in the navigation plan
            return None
        else:
            return search_result[0][0]

    def get_next_road(self, plan_index: int) -> Union[None, int]:
        """
        Given a road_id, returns the next road_id in the navigation plan
        :param plan_index: the index of the road in plan
        :return: the next road_id in the navigation plan, None if the road has no next or does not exist in the plan
        """
        if plan_index < len(self.road_ids):
            next_road_id = self.road_ids[plan_index + 1]
            return next_road_id
        else:
            return None

    def get_previous_road(self, plan_index: int) -> int:
        """
        Given a road_id, returns the previous road_id in the navigation plan
        :param plan_index: the index of the road in plan
        :return: the next road_id in the navigation plan, None if the road_id has no prev or does not exist in the plan
        """
        if plan_index > 0
            prev_road_id = self.road_ids[plan_index - 1]
            return prev_road_id
        else:
            return None

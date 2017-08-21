from typing import Union

import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class NavigationPlanMsg(DDSTypedMsg):
    """
        This class hold the navigation plan.
    """
    def __init__(self, road_ids: np.array):
        """
        Initialization of the navigation plan. This is an initial implementation which contains only a list o road ids.
        :param road_ids: list of road ids corresponding to the map.
        """
        self.road_ids = road_ids

    def get_current_road_index(self, road_id: int) -> Union[None, int]:
        """
        Given a road_id, returns the index of this road_id in the navigation plan
        :param road_id: a road_id as exists in MapAPI
        :return: the index of the road_id in the navigation plan, None if road_id does not exist in the plan
        """
        pass

    def get_next_road_id(self, road_id: int) -> Union[None, int]:
        """
        Given a road_id, returns the next road_id in the navigation plan
        :param road_id: a road_id as exists in MapAPI
        :return: the next road_id in the navigation plan, None if the road_id has no next or does not exist in the plan
        """
        pass

    def get_previous_road_id(self, road_id: int) -> int:
        """
        Given a road_id, returns the previous road_id in the navigation plan
        :param road_id: a road_id as exists in MapAPI
        :return: the next road_id in the navigation plan, None if the road_id has no prev or does not exist in the plan
        """
        pass

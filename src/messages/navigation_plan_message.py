from logging import Logger
from typing import Union
import numpy as np
from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg


class NavigationPlanMsg(DDSNonTypedMsg):
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
        self.road_ids = np.array(road_ids)

    def __get_road_index_in_plan(self, road_id: int, logger: Logger) -> Union[None, int]:
        """
        Given a road_id, returns the index of this road_id in the navigation plan
        :param road_id:
        :param logger: for logging
        :return: index of road_id in the plan
        """
        search_result = np.where(self.road_ids == road_id)
        if len(search_result[0]) == 0:  # target road_id not found in the navigation plan
            logger.warning("Navigation: Can't find get road_id %d in plan", road_id)
            return None
        else:
            return search_result[0][0]

    def get_next_road(self, road_id: int, logger: Logger) -> Union[None, int]:
        """
        Given a road_id, returns the next road_id in the navigation plan
        :param road_id: the current road_id
        :param logger: for logging
        :return: the next road_id in the navigation plan, None if the road has no next or does not exist in the plan
        """
        if road_id is None:
            return None
        plan_index = self.__get_road_index_in_plan(road_id, logger)
        if plan_index is None:
            logger.warning("Navigation: Can't get find road_in=%d in plan", road_id)
            return None

        if plan_index < len(self.road_ids) - 1:
            next_road_id = self.road_ids[plan_index + 1]
            return next_road_id
        else:
            if plan_index < len(self.road_ids):
                logger.warning("Navigation: Can't get next road in plan. Current road: %d", self.road_ids[plan_index])
            else:
                logger.warning("Navigation: Can't get current road in plan. Current plan_index: %d", plan_index)
            return None

    def get_previous_road(self, road_id: int, logger: Logger) -> Union[None, int]:
        """
        Given a road_id, returns the previous road_id in the navigation plan
        :param logger: for logging
        :param road_id: the current road_id
        :return: the previous road_id in the navigation plan, None if the road_id has no prev or does not exist in the plan
        """
        if road_id is None:
            return None
        plan_index = self.__get_road_index_in_plan(road_id, logger)
        if plan_index > 0:
            prev_road_id = self.road_ids[plan_index - 1]
            return prev_road_id
        else:
            if plan_index >= 0:
                logger.warning("Navigation: Can't get previous road in plan. Current road: %d", self.road_ids[plan_index])
            else:
                logger.warning("Navigation: Can't get current road in plan. Current plan_index: %d", plan_index)
            return None

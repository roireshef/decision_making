from typing import Optional

import numpy as np

from common_data.interface.py.idl_generated_files.dm import LcmNavigationPlan
from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNonTypedIntNumpyArray import LcmNonTypedIntNumpyArray
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from mapping.src.exceptions import RoadNotFound, raises


class NavigationPlanMsg(PUBSUB_MSG_IMPL):
    """
        This class hold the navigation plan.
        It also implements function (required by MapAPI) that iterate over the roads list in the navigation plan.
        Important assumption: we assume that road_ids is a UNIQUE list, containing each value only once.
    """
    def __init__(self, road_ids):
        # type: (np.ndarray) -> None
        """
        Initialization of the navigation plan. This is an initial implementation which contains only a list o road ids.
        :param road_ids: Numpy array (dtype has to be int) of road ids corresponding to the map.
        """
        self.road_ids = road_ids.astype(np.int)

    def serialize(self):
        # type: () -> LcmNavigationPlan
        lcm_msg = LcmNavigationPlan()

        # TODO: This solves inconsistency between mock and real lcm, since lcm cpp code treats the array as floats.
        # TODO: cont - need to change if causes time delays.
        lcm_msg.road_ids = SerializationUtils.serialize_non_typed_int_array(self.road_ids)
        lcm_msg.road_ids.type = ""

        lcm_msg.type = ""

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmNavigationPlan) -> NavigationPlanMsg
        return cls(SerializationUtils.deserialize_any_array(lcmMsg.road_ids))

    @raises(RoadNotFound)
    def get_road_index_in_plan(self, road_id, start=None, end=None):
        # type:  (int, Optional[int], Optional[int]) -> int
        """
        Given a road_id, returns the index of this road_id in the navigation plan
        :param road_id: the request road_id to look for in the plan
        :param start: optional. starting index to look from in the plan (inclusive)
        :param end: optional. ending index to look up to in the plan (inclusive)
        :return: index of road_id in the plan
        """
        try:
            if start is None:
                start = 0
            if end is None:
                end = len(self.road_ids)
            return np.where(self.road_ids[start:(end+1)] == road_id)[0][0] + start
        except IndexError:
            raise RoadNotFound("Road ID {} is not in clipped (indices: [{}, {}]) plan's road-IDs [{}]"
                               .format(road_id, start, end, self.road_ids[start:(end+1)]))

    @raises(RoadNotFound)
    def get_next_road(self, road_id):
        # type:  (int) -> int
        """
        Given a road_id, returns the next road_id in the navigation plan
        :param road_id: the current road_id
        :return: the next road_id in the navigation plan, None if the road has no next or does not exist in the plan
        """
        try:
            plan_index = self.get_road_index_in_plan(road_id)
            next_road_id = self.road_ids[plan_index + 1]
            return next_road_id
        except (IndexError, RoadNotFound) as e:
            raise RoadNotFound("Navigation: Can't find the next road_id to #" + str(road_id) + " in plan " +
                               str(self.road_ids) + ". " + str(e))

    @raises(RoadNotFound)
    def get_previous_road(self, road_id):
        # type:  (int) -> int
        """
        Given a road_id, returns the previous road_id in the navigation plan
        :param road_id: the current road_id
        :return: the previous road_id in the navigation plan, None if the road_id has no prev or does not exist in the plan
        """
        try:
            plan_index = self.get_road_index_in_plan(road_id)
            prev_road_id = self.road_ids[plan_index - 1]
            return prev_road_id
        except (IndexError, RoadNotFound) as e:
            raise RoadNotFound("Navigation: Can't find the previous road_id to #" + str(road_id) + " in plan " +
                               str(self.road_ids) + ". " + str(e))


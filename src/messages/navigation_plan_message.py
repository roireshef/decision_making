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

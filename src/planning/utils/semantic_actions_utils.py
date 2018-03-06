import numpy as np
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    LATERAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.state.state import ObjectSize


class SemanticActionsUtils:

    @staticmethod
    def get_ego_lon_margin(ego_size: ObjectSize) -> float:
        """
        calculate margin for a safe longitudinal distance between ego and another car based on ego length
        :param ego_size: the size of ego (length, width, height)
        :return: safe longitudinal margin
        """
        return ego_size.length / 2 + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

    @staticmethod
    def get_ego_lat_margin(ego_size: ObjectSize) -> float:
        """
        calculate margin for a safe lateral distance between ego and another car based on ego width
        :param ego_size: the size of ego (length, width, height)
        :return: safe lateral margin
        """
        return ego_size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT

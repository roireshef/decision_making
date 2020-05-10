from decision_making.src.exceptions import raises, LaneAttributeNotFound
from decision_making.src.global_constants import LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD
from decision_making.src.messages.scene_static_enums import LaneMappingStatusType, \
    MapLaneDirection, GMAuthorityType, LaneConstructionType
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase


class RouteUtils:

    class DrivableLaneIndicatorMapping:
        @staticmethod
        def mapping_status_based_drivablity_indicator(mapping_status_attribute: LaneMappingStatusType) -> float:
            """
            Determines if the lane is drivable based on mapping status
            :param mapping_status_attribute: type of mapped
            :return: True if lane is considered drivable, false otherwise
            """
            if ((mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
                    (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
                return True
            return False

        @staticmethod
        def construction_zone_based_drivablity_indicator(construction_zone_attribute: LaneConstructionType) -> float:
            """
            Determines if the lane is drivable based on construction zone status
            :param construction_zone_attribute: type of lane construction
            :return: True if lane is considered drivable, false otherwise
            """
            if ((construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
                    (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
                return True
            return False

        @staticmethod
        def lane_dir_in_route_based_drivablity_indicator(lane_dir_in_route_attribute: MapLaneDirection) -> float:
            """
            Determines if the lane is drivable based on lane directions relative to host vehicle
            :param lane_dir_in_route_attribute: map lane direction in respect to host
            :return: True if lane is considered drivable, false otherwise
            """
            if ((lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
                    (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                    (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
                return True
            return False

        @staticmethod
        def gm_authority_based_drivablity_indicator(gm_authority_attribute: GMAuthorityType) -> float:
            """
            Determines if the lane is drivable based on GM authorized driving area
            :param gm_authority_attribute: type of GM authority
            :return: True if lane is considered drivable, false otherwise
            """
            if gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None:
                return True
            return False

    # This is a class variable of RouteUtils, containing all indicator methods for a lane being drivable or not
    drivable_lane_indicator_methods = {LaneMappingStatusType: DrivableLaneIndicatorMapping.mapping_status_based_drivablity_indicator,
                                       GMAuthorityType: DrivableLaneIndicatorMapping.gm_authority_based_drivablity_indicator,
                                       LaneConstructionType: DrivableLaneIndicatorMapping.construction_zone_based_drivablity_indicator,
                                       MapLaneDirection: DrivableLaneIndicatorMapping.lane_dir_in_route_based_drivablity_indicator}

    @staticmethod
    @raises(LaneAttributeNotFound)
    def is_lane_segment_drivable(lane_segment_base_data: SceneLaneSegmentBase) -> bool:
        """
        Determines if a lane segment is drivable
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: A boolean indicating if the lane is drivable or not
        """
        # Now iterate over all the active lane attributes for the lane segment
        for lane_attribute_index in lane_segment_base_data.a_i_active_lane_attribute_indices:
            # lane_attribute_index gives the index lookup for lane attributes and confidences
            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attributes):
                lane_attribute = lane_segment_base_data.a_cmp_lane_attributes[lane_attribute_index]
            else:
                raise LaneAttributeNotFound('Lane_attribute_index {0} doesn\'t have corresponding lane attribute value'
                                            .format(lane_attribute_index))

            if lane_attribute_index < len(lane_segment_base_data.a_cmp_lane_attribute_confidences):
                lane_attribute_confidence = lane_segment_base_data.a_cmp_lane_attribute_confidences[lane_attribute_index]
            else:
                raise LaneAttributeNotFound('Lane_attribute_index {0} doesn\'t have corresponding lane attribute '
                                            'confidence value'.format(lane_attribute_index))

            if lane_attribute_confidence < LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD:
                continue

            try:
                is_lane_drivable = RouteUtils.drivable_lane_indicator_methods[type(lane_attribute)](lane_attribute)
            except KeyError:
                raise LaneAttributeNotFound(f"Could not find the drivable lane indicator method that corresponds to lane attribute "
                                            f"type {type(lane_attribute)}. The supported types are "
                                            f"{RouteUtils.drivable_lane_indicator_methods.keys()}.")

            if not is_lane_drivable:
                return False

        return True

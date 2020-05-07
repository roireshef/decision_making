from decision_making.src.exceptions import raises, LaneAttributeNotFound
from decision_making.src.global_constants import LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD
from decision_making.src.messages.scene_static_enums import LaneMappingStatusType, \
    MapLaneDirection, GMAuthorityType, LaneConstructionType
from decision_making.src.messages.scene_static_message import SceneLaneSegmentBase


class RouteUtils:

    class GMFAIndicatorMapping:
        @staticmethod
        def mapping_status_based_gmfa_indicator(mapping_status_attribute: LaneMappingStatusType) -> float:
            """
            Determines if GMFA based on mapping status
            :param mapping_status_attribute: type of mapped
            :return: True if lane is considered GMFA, false otherwise
            """
            if ((mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap) or
                    (mapping_status_attribute == LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_MDMap)):
                return False
            return True

        @staticmethod
        def construction_zone_based_gmfa_indicator(construction_zone_attribute: LaneConstructionType) -> float:
            """
            Determines if GMFA based on construction zone status
            :param construction_zone_attribute: type of lane construction
            :return: True if lane is considered GMFA, false otherwise
            """
            if ((construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Normal) or
                    (construction_zone_attribute == LaneConstructionType.CeSYS_e_LaneConstructionType_Unknown)):
                return False
            return True

        @staticmethod
        def lane_dir_in_route_based_gmfa_indicator(lane_dir_in_route_attribute: MapLaneDirection) -> float:
            """
            Determines if GMFA based on lane directions relative to host vehicle
            :param lane_dir_in_route_attribute: map lane direction in respect to host
            :return: True if lane is considered GMFA, false otherwise
            """
            if ((lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle) or
                    (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle) or
                    (lane_dir_in_route_attribute == MapLaneDirection.CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle)):
                return False
            return True

        @staticmethod
        def gm_authority_based_gmfa_indicator(gm_authority_attribute: GMAuthorityType) -> float:
            """
            Determines if GMFA based on GM authorized driving area
            :param gm_authority_attribute: type of GM authority
            :return: True if lane is considered GMFA, false otherwise
            """
            if gm_authority_attribute == GMAuthorityType.CeSYS_e_GMAuthorityType_None:
                return False
            return True

    # This is a class variable that is shared between all RoutePlanner instances
    gmfa_indicator_methods = {LaneMappingStatusType: GMFAIndicatorMapping.mapping_status_based_gmfa_indicator,
                              GMAuthorityType: GMFAIndicatorMapping.gm_authority_based_gmfa_indicator,
                              LaneConstructionType: GMFAIndicatorMapping.construction_zone_based_gmfa_indicator,
                              MapLaneDirection: GMFAIndicatorMapping.lane_dir_in_route_based_gmfa_indicator}

    @staticmethod
    @raises(LaneAttributeNotFound)
    def is_lane_segment_gmfa(lane_segment_base_data: SceneLaneSegmentBase) -> bool:
        """
        Determines if a lane segment is considered to be a GMFA
        :param lane_segment_base_data: SceneLaneSegmentBase for the concerned lane
        :return: A boolean indicating if the lane is a GMFA area
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
                is_lane_considered_gmfa = RouteUtils.gmfa_indicator_methods[type(lane_attribute)](lane_attribute)
            except KeyError:
                raise LaneAttributeNotFound(f"Could not find the GMFA indicator method that corresponds to lane attribute "
                                            f"type {type(lane_attribute)}. The supported types are "
                                            f"{RouteUtils.gmfa_indicator_methods.keys()}.")

            if is_lane_considered_gmfa:
                return True

        return False

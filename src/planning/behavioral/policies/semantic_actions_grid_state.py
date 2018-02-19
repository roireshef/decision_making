from logging import Logger

from decision_making.src.global_constants import EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.global_constants import SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LON_REAR, LON_MARGIN_FROM_EGO
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.policies.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticBehavioralState, \
    RoadSemanticOccupancyGrid, LON_CELL
from decision_making.src.state.state import EgoState, State, DynamicObject
from mapping.src.service.map_service import MapService


class SemanticActionsGridState(SemanticBehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        super().__init__(road_occupancy_grid=road_occupancy_grid)
        self.ego_state = ego_state

    @classmethod
    def create_from_state(cls, state: State, logger: Logger):
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest to ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :return: road semantic occupancy grid
        """

        ego_state = state.ego_state
        dynamic_objects = state.dynamic_objects

        default_navigation_plan = MapService.get_instance().get_road_based_navigation_plan(
            current_road_id=ego_state.road_localization.road_id)

        ego_lane = ego_state.road_localization.lane_num

        # Generate grid cells
        semantic_occupancy_dict: RoadSemanticOccupancyGrid = dict()
        # We take into consideration only the current and adjacent lanes
        relative_lane_keys = [-1, 0, 1]
        num_lanes_in_road = 2 # TODO (hack for demo) MapService.get_instance().get_road(state.ego_state.road_localization.road_id).lanes_num
        # TODO: treat objects out of road
        filtered_absolute_lane_keys = list(
            filter(lambda relative_lane: 0 <= ego_lane + relative_lane < num_lanes_in_road, relative_lane_keys))

        optional_lon_keys = [SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR]
        for lon_key in optional_lon_keys:
            for lane_key in filtered_absolute_lane_keys:
                occupancy_index = (lane_key, lon_key)
                semantic_occupancy_dict[occupancy_index] = []

        # Allocate dynamic objects
        for dynamic_object in dynamic_objects:

            dist_from_object_rear_to_ego_front, dist_from_ego_rear_to_object_front = \
                SemanticActionsGridState.calc_relative_distances(state, default_navigation_plan, dynamic_object)
            object_relative_lane = dynamic_object.road_localization.lane_num - ego_lane

            # Determine cell index in occupancy grid (lane and longitudinal location), under the following assumptions:
            # 1. We filter out objects that are more than one lane away from ego
            # 2. Longitudinal location is defined by the grid structure:
            #       - front cells: starting from ego front + LON_MARGIN_FROM_EGO [m] and forward
            #       - back cells: starting from ego back - LON_MARGIN_FROM_EGO[m] and backwards
            if object_relative_lane == 0 or object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if dist_from_object_rear_to_ego_front > LON_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_FRONT)

                elif dist_from_ego_rear_to_object_front < -LON_MARGIN_FROM_EGO:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_REAR)

                else:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_SAME)
            else:
                continue

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index in semantic_occupancy_dict:

                # We treat the object only if its distance is smaller than the distance we
                # would have travelled for the planning horizon in the average speed between current and target vel.
                if dist_from_object_rear_to_ego_front > SemanticActionsUtils.compute_distance_by_velocity_diff(ego_state.v_x):
                    continue

                if len(semantic_occupancy_dict[occupancy_index]) == 0:
                    # add to occupancy grid
                    semantic_occupancy_dict[occupancy_index].append(dynamic_object)
                else:
                    # get first object in list of objects in the cell as reference
                    object_in_cell = semantic_occupancy_dict[occupancy_index][0]

                    dist_from_grid_object_rear_to_ego_front, dist_from_ego_rear_to_grid_object_front = \
                        SemanticActionsGridState.calc_relative_distances(state, default_navigation_plan, object_in_cell)

                    if occupancy_index[LON_CELL] == SEMANTIC_CELL_LON_FRONT:
                        # take the object with least lon
                        if dist_from_object_rear_to_ego_front < dist_from_grid_object_rear_to_ego_front:
                            # replace first object the the closer one
                            semantic_occupancy_dict[occupancy_index][0] = dynamic_object
                    else:
                        # Assumption - taking the object with the largest lon even in the ASIDE cells
                        if dist_from_ego_rear_to_object_front > dist_from_ego_rear_to_grid_object_front:
                            # replace first object the the closer one
                            semantic_occupancy_dict[occupancy_index][0] = dynamic_object

        return cls(semantic_occupancy_dict, ego_state)

    @staticmethod
    def calc_relative_distances(state: State,
                                default_navigation_plan: NavigationPlanMsg,
                                object_in_cell: DynamicObject) -> [float, float]:
        """
        Given dynamic object in a cell, calculate the distance from the object's rear to ego front and
          from ego rear to the object's front
        :param state: the State
        :param default_navigation_plan:
        :param object_in_cell: dynamic object in some cell
        :return: two distances: distance from the object's rear to ego front and from ego rear to the object's front
        """
        ego_state = state.ego_state
        object_relative_localization = MapService.get_instance().compute_road_localizations_diff(
                        reference_localization=ego_state.road_localization,
                        object_localization=object_in_cell.road_localization,
                        navigation_plan=default_navigation_plan
                    )
        object_lon_dist = object_relative_localization.rel_lon
        dist_from_object_rear_to_ego_front = object_lon_dist - object_in_cell.size.length / 2 - \
                                             (ego_state.size.length - EGO_ORIGIN_LON_FROM_REAR)
        dist_from_ego_rear_to_object_front = object_lon_dist + object_in_cell.size.length / 2 + \
                                             EGO_ORIGIN_LON_FROM_REAR
        return dist_from_object_rear_to_ego_front, dist_from_ego_rear_to_object_front

from logging import Logger

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.planning.behavioral.constants import SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LON_REAR, LON_MARGIN_FROM_EGO, BEHAVIORAL_PLANNING_HORIZON
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticBehavioralState, \
    RoadSemanticOccupancyGrid, LON_CELL
from decision_making.src.state.state import EgoState, State
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
        optional_lane_keys = [-1, 0, 1]
        lanes_in_road = MapService.get_instance().get_road(state.ego_state.road_localization.road_id).lanes_num
        filtered_lane_keys = list(
            filter(lambda relative_lane: 0 <= ego_lane + relative_lane < lanes_in_road, optional_lane_keys))

        optional_lon_keys = [SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR]
        for lon_key in optional_lon_keys:
            for lane_key in filtered_lane_keys:
                occupancy_index = (lane_key, lon_key)
                semantic_occupancy_dict[occupancy_index] = []

        # Allocate dynamic objects
        for dynamic_object in dynamic_objects:
            object_relative_localization = MapService.get_instance().compute_road_localizations_diff(
                reference_localization=ego_state.road_localization,
                object_localization=dynamic_object.road_localization,
                navigation_plan=default_navigation_plan
            )
            object_lon_dist = object_relative_localization.rel_lon
            object_dist_from_front = object_lon_dist - ego_state.size.length
            object_relative_lane = int(dynamic_object.road_localization.lane_num - ego_lane)

            # Determine cell index in occupancy grid (lane and longitudinal location), under the following assumptions:
            # 1. We filter out objects that are at more one distant from ego
            # 2. Longitudinal location is defined by the grid structure:
            #       - front cells: starting from ego front + LON_MARGIN_FROM_EGO [m] and forward
            #       - back cells: starting from ego back - LON_MARGIN_FROM_EGO[m] and backwards
            if object_relative_lane == 0 or object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if object_dist_from_front > LON_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_FRONT)

                elif object_lon_dist > -1 * LON_MARGIN_FROM_EGO:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_SAME)

                else:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_REAR)
            else:
                continue

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            # TODO: treat objects out of road
            if occupancy_index in semantic_occupancy_dict:

                # We treat the object only if its distance is equal to the to the distance we
                # would have travelled for the planning horizon in the average speed between current and target vel.
                if object_dist_from_front > BEHAVIORAL_PLANNING_HORIZON * \
                        0.5 * (ego_state.v_x + BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT):
                    continue

                if len(semantic_occupancy_dict[occupancy_index]) == 0:
                    # add to occupancy grid
                    semantic_occupancy_dict[occupancy_index].append(dynamic_object)
                else:
                    # get first objects in list of objects in cell as reference
                    object_in_cell = semantic_occupancy_dict[occupancy_index][0]
                    object_in_grid_lon_dist = MapService.get_instance().compute_road_localizations_diff(
                        reference_localization=ego_state.road_localization,
                        object_localization=object_in_cell.road_localization,
                        navigation_plan=default_navigation_plan
                    ).rel_lon

                    object_in_grid_dist_from_front = object_in_grid_lon_dist - ego_state.size.length

                    if occupancy_index[LON_CELL] == SEMANTIC_CELL_LON_FRONT:
                        # take the object with least lon
                        if object_lon_dist < object_in_grid_dist_from_front:
                            # replace first object the the closer one
                            semantic_occupancy_dict[occupancy_index][0] = dynamic_object
                    else:
                        # Assumption - taking the object with the largest long even in the ASIDE cells
                        # take the object with largest lon
                        if object_lon_dist > object_in_grid_dist_from_front:
                            # replace first object the the closer one
                            semantic_occupancy_dict[occupancy_index][0] = dynamic_object

        return cls(semantic_occupancy_dict, ego_state)
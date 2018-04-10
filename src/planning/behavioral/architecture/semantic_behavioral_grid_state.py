import numpy as np
from logging import Logger

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_LOOKAHEAD_DIST
from decision_making.src.global_constants import SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LON_REAR, LON_MARGIN_FROM_EGO
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.policies.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticBehavioralState, \
    RoadSemanticOccupancyGrid, LON_CELL
from decision_making.src.planning.types import FP_SX, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import EgoState, State, DynamicObject
from mapping.src.service.map_service import MapService


class SemanticBehavioralGridState(SemanticBehavioralState):
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

        # We filter out objects that are more than one lane away from ego (consider only the current and adjacent lanes)
        filtered_dynamic_objects = filter(lambda x: (x.road_localization.lane_num-ego_lane) in (-1, 0, 1), dynamic_objects)

        # We treat the object only if its distance is smaller than the distance we
        # would have travelled for the planning horizon in the average speed between current and target vel.
        maximal_considered_distance = min(BEHAVIORAL_PLANNING_LOOKAHEAD_DIST, SemanticActionsUtils.
                                          compute_distance_by_velocity_diff(ego_state.v_x))

        # Allocate dynamic objects
        for dynamic_object in filtered_dynamic_objects:

            dist_from_object_rear_to_ego_front, dist_from_ego_rear_to_object_front = \
                SemanticBehavioralGridState.calc_relative_distances(state, default_navigation_plan, dynamic_object)
            object_relative_lane = dynamic_object.road_localization.lane_num - ego_lane

            if dist_from_object_rear_to_ego_front > maximal_considered_distance:
                continue

            # Determine cell index in occupancy grid (lane and longitudinal location), under the assumption:
            #   Longitudinal location is defined by the grid structure:
            #       - front cells: starting from ego front + LON_MARGIN_FROM_EGO [m] and forward
            #       - back cells: starting from ego back - LON_MARGIN_FROM_EGO[m] and backwards

            # Object is one lane on the left/right and not very far
            if dist_from_object_rear_to_ego_front > LON_MARGIN_FROM_EGO:
                # Object in front of vehicle
                occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_FRONT)

            elif dist_from_ego_rear_to_object_front < -LON_MARGIN_FROM_EGO:
                # Object behind rear of vehicle
                occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_REAR)

            else:
                # Object vehicle aside of ego
                occupancy_index = (object_relative_lane, SEMANTIC_CELL_LON_SAME)

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index in semantic_occupancy_dict:

                # get first object in list of objects in the cell as reference
                object_in_cell = semantic_occupancy_dict[occupancy_index][0]

                dist_from_grid_object_rear_to_ego_front, dist_from_ego_rear_to_grid_object_front = \
                    SemanticBehavioralGridState.calc_relative_distances(state, default_navigation_plan, object_in_cell)

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
            else:
                # dynamic_object is the first one seen in this occupancy cell
                semantic_occupancy_dict[occupancy_index] = [dynamic_object]

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
        :return: two relative longitudes:
        1. longitude of object's rear relatively to ego front
        2. longitude of object's front relatively to ego rear
        Only one of the longitudes is relevant, depending on the sign of object's relative longitude.
        """

        # TODO: the relative localization calculated here assumes that all objects are located on the same road.
        # TODO: Fix after demo and calculate longitudinal difference properly in the general case
        ego_state = state.ego_state
        road_id = ego_state.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)
        road_frenet = FrenetSerret2DFrame(road_points)

        # Object Frenet state
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([object_in_cell.x, object_in_cell.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            object_in_cell.x, object_in_cell.y,
            object_in_cell.yaw,
            object_in_cell.v_x,
            object_in_cell.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Ego Frenet state
        ego_init_cstate = np.array(
            [ego_state.x, ego_state.y, ego_state.yaw, ego_state.v_x, ego_state.acceleration_lon, ego_state.curvature])
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        # Relative longitudinal distance
        object_relative_lon = obj_init_fstate[FS_SX] - ego_init_fstate[FS_SX]

        # the following two distances are signed
        lon_obj_rear_from_ego_front = object_relative_lon - (object_in_cell.size.length/2 + ego_state.size.length/2)
        lon_obj_front_from_ego_rear = object_relative_lon + (object_in_cell.size.length/2 + ego_state.size.length/2)
        return lon_obj_rear_from_ego_front, lon_obj_front_from_ego_rear

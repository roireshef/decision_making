import numpy as np
from logging import Logger
from typing import List

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, \
    SemanticBehavioralState, RoadSemanticOccupancyGrid, SemanticActionSpec, SemanticAction, SEMANTIC_CELL_LANE
from decision_making.src.state.state import EgoState, State, DynamicObject
from mapping.src.model.map_api import MapAPI

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


MIN_OVERTAKE_VEL = 2  # [m/s]
MIN_CHANGE_LANE_TIME = 4  # sec
# LONG_ACHEIVING_TIME = 10  # sec


class NovDemoBehavioralState(SemanticBehavioralState):
    def __init__(self, road_occupancy_grid: RoadSemanticOccupancyGrid, ego_state: EgoState):
        super().__init__(road_occupancy_grid=road_occupancy_grid)
        self.ego_state = ego_state


    @classmethod
    def create_from_state(cls, state: State, map_api: MapAPI, logger: Logger):
        """
        :return: a new and updated BehavioralState
        """
        semantic_occupancy_grid = cls._generate_semantic_occupancy_grid(ego_state=state.ego_state,
                                                                        dynamic_objects=state.dynamic_objects,
                                                                        map_api=map_api,
                                                                        logger=logger)

        return NovDemoBehavioralState(road_occupancy_grid=semantic_occupancy_grid, ego_state=state.ego_state)

    @staticmethod
    def _generate_semantic_occupancy_grid(ego_state: EgoState, dynamic_objects: List[
        DynamicObject], map_api: MapAPI, logger: Logger) -> RoadSemanticOccupancyGrid:
        """
        Occupy the occupancy grid.
        This method iterates over all dynamic objects, and fits them into the relevant cell
        in the semantic occupancy grid (semantic_lane, semantic_lon).
        Each cell holds a list of objects that are within the cell borders.
        In this particular implementation, we keep up to one dynamic object per cell, which is the closest ego.
         (e.g. in the cells in front of ego, we keep objects with minimal longitudinal distance
         relative to ego front, while in all other cells we keep the object with the maximal longitudinal distance from
         ego front).
        :param ego_state:
        :param dynamic_objects:
        :return: road semantic occupancy grid
        """

        default_navigation_plan = map_api.get_road_based_navigation_plan(
            current_road_id=ego_state.road_localization.road_id)

        ego_lane = ego_state.road_localization.lane_num

        semantic_occupancy_dict: RoadSemanticOccupancyGrid = dict()
        for dynamic_object in dynamic_objects:

            object_relative_localization = dynamic_object.get_relative_road_localization(
                ego_road_localization=ego_state.road_localization, ego_nav_plan=default_navigation_plan,
                map_api=map_api, logger=logger)
            object_lon_dist = object_relative_localization.rel_lon
            object_dist_from_front = object_lon_dist - ego_state.size.length
            object_relative_lane = int(dynamic_object.road_localization.lane_num - ego_lane)

            # Determine cell index in occupancy grid
            if object_relative_lane == 0:
                # Object is on same lane as ego
                if object_dist_from_front > 0.0:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                else:
                    # Object behind vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            elif object_relative_lane == 1 or object_relative_lane == -1:
                # Object is one lane on the left/right

                if object_dist_from_front > SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object in front of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_FRONT)

                elif object_lon_dist > -1 * SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO:
                    # Object vehicle aside of ego
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_ASIDE)

                else:
                    # Object behind rear of vehicle
                    occupancy_index = (object_relative_lane, SEMANTIC_GRID_BEHIND)

            # Add object to occupancy grid
            # keeping only a single dynamic object per cell. List is used for future dev.
            if occupancy_index not in semantic_occupancy_dict:
                # add to occupancy grid
                semantic_occupancy_dict[occupancy_index] = [dynamic_object]
            else:
                object_in_cell = semantic_occupancy_dict[occupancy_index][0]
                object_in_grid_lon_dist = object_in_cell.get_relative_road_localization(
                    ego_road_localization=ego_state.road_localization,
                    ego_nav_plan=default_navigation_plan,
                    map_api=map_api, logger=logger).rel_lon
                object_in_grid_dist_from_front = object_in_grid_lon_dist - ego_state.size.length

                if occupancy_index[1] == SEMANTIC_GRID_FRONT:
                    # take the object with least lon
                    if object_lon_dist < object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object
                else:
                    # Assumption - taking the object with the largest long even in the ASIDE cells
                    # take the object with largest lon
                    if object_lon_dist > object_in_grid_dist_from_front:
                        # replace object the the closer one
                        semantic_occupancy_dict[occupancy_index][0] = dynamic_object

        return semantic_occupancy_dict

class NovDemoPolicy(SemanticActionsPolicy):

    def _eval_actions(self, behavioral_state: NovDemoBehavioralState, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
        """
        Evaluate the generated actions using the full state.
        Gets a list of actions to evaluate so and returns a vector representing their costs.
        A set of actions is provided, enabling assessing them dependently.
        Note: the semantic actions were generated using the behavioral state which isn't necessarily captures
         all relevant details in the scene. Therefore the evaluation is done using the full state.
        :param state: world state
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions
        """

        straight_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 0)
        left_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, 1)
        right_lane_ind = NovDemoPolicy._get_action_ind_by_lane(semantic_actions, actions_spec, -1)

        desired_vel = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT

        right_is_fast = right_lane_ind is not None and desired_vel - actions_spec[right_lane_ind].v < MIN_OVERTAKE_VEL
        right_is_occupied = len(behavioral_state.road_occupancy_grid[(-1, 0)]) > 0
        straight_is_fast = straight_lane_ind is not None and \
                           desired_vel - actions_spec[straight_lane_ind].v < MIN_OVERTAKE_VEL
        left_is_faster = left_lane_ind is not None and (straight_lane_ind is None or
                            actions_spec[left_lane_ind].v - actions_spec[straight_lane_ind].v >= MIN_OVERTAKE_VEL)
        left_is_occupied = len(behavioral_state.road_occupancy_grid[(1, 0)]) > 0

        costs = np.zeros(len(semantic_actions))
        # move right if both straight and right lanes are fast
        if right_is_fast and (straight_is_fast or straight_lane_ind is None) and not right_is_occupied:
            costs[right_lane_ind] = 1.
            return costs
        # move left if straight is slow and the left is faster than straight
        if not straight_is_fast and (left_is_faster or straight_lane_ind is None) and not left_is_occupied:
            costs[left_lane_ind] = 1.
            return costs
        costs[straight_lane_ind] = 1.
        return costs

        # right_is_very_far = right_lane_ind is not None and actions_spec[right_lane_ind].t > LONG_ACHEIVING_TIME
        # straight_is_far = straight_lane_ind is not None and actions_spec[straight_lane_ind].t > MIN_CHANGE_LANE_TIME
        # left_is_very_far = left_lane_ind is not None and actions_spec[left_lane_ind].t > LONG_ACHEIVING_TIME
        # if (right_is_fast or right_is_very_far) and \
        #         (straight_is_fast or straight_is_far or straight_lane_ind is None) and \
        #         not right_is_occupied:
        #     costs[right_lane_ind] = 1.
        #     return costs
        # # move left if straight is slow and close and the left is far or faster than straight
        # if not (straight_is_fast or straight_is_far) and \
        #         (left_is_faster or left_is_very_far or straight_lane_ind is None) and \
        #         not left_is_occupied:
        #     costs[left_lane_ind] = 1.
        #     return costs
        # costs[straight_lane_ind] = 1.
        # return costs

    @staticmethod
    def _get_action_ind_by_lane(semantic_actions: List[SemanticAction], actions_spec: List[SemanticActionSpec],
                                cell_lane: int):
        action_ind = [i for i, action in enumerate(semantic_actions) if action.cell[SEMANTIC_CELL_LANE] == cell_lane]
        # verify that change lane time is large enough
        if len(action_ind) > 0 and (cell_lane == 0 or actions_spec[action_ind[0]].t >= MIN_CHANGE_LANE_TIME):
            return action_ind[0]
        else:
            return None

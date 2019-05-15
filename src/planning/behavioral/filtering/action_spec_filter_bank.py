import numpy as np
from typing import List

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    SAFETY_HEADWAY, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, BIG_EPS
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX
from decision_making.src.planning.types import LAT_CELL
from decision_making.src.planning.types import FS_SA, FS_DX, LIMIT_MIN, C_V
from decision_making.src.planning.types import FS_SX, FS_SV, LAT_CELL
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t) for action_spec in action_specs]


class FilterForKinematics(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean list per action spec: True if a spec passed the filter
        """
        # extract all relevant information for boundary conditions
        relative_lanes = np.array([spec.relative_lane for spec in action_specs])
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])

        # represent initial and terminal boundary conditions (for two Frenet axes s,d)
        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA+1)], terminal_fstates[:, :(FS_SA+1)]), axis=1)
        constraints_d = np.concatenate((initial_fstates[:, FS_DX:], terminal_fstates[:, FS_DX:]), axis=1)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        T = np.array([spec.t for spec in action_specs])
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T))

        # solve for s(t) and d(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        are_valid = []

        # debug
        velocities = initial_fstates[:, FS_SV]
        for poly_s, poly_d, t, lane, spec, iv in zip(poly_coefs_s, poly_coefs_d, T, relative_lanes, action_specs, velocities):
            # TODO: in the future, consider leaving only a single action (for better "learnability")
            if spec.only_padding_mode:
                are_valid.append(True)
                continue
            if spec.v > BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED:
                are_valid.append(False)
                continue
            # extract the relevant (cached) frenet frame per action according to the destination lane
            frenet_frame = behavioral_state.extended_lane_frames[lane]

            # if the action is static, there's a chance the 5th order polynomial is actually a degenerate one
            # (has lower degree), so we clip the first zero coefficients and send a polynomial with lower degree
            # TODO: This handling of polynomial coefficients being 5th or 4th order should happen in an inner context and get abstracted from this method
            first_non_zero = np.argmin(np.equal(poly_s, 0)) if isinstance(spec.recipe, StaticActionRecipe) else 0
            is_valid_in_frenet = KinematicUtils.filter_by_longitudinal_frenet_limits(poly_s[np.newaxis, first_non_zero:], np.array([t]),
                                                                                     LON_ACC_LIMITS, np.array([0, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED]), frenet_frame.s_limits)[0]
            # frenet checks are analytical and do not require conversions so they are faster. If they do not pass,
            # we can save time by not checking cartesian limits
            if not is_valid_in_frenet:
                are_valid.append(False)
                continue

            total_time = max(MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, t)
            time_samples = np.arange(0, total_time + EPS, WERLING_TIME_RESOLUTION)

            # generate a SamplableWerlingTrajectory (combination of s(t), d(t) polynomials applied to a Frenet frame)
            samplable_trajectory = SamplableWerlingTrajectory(0, t, t, total_time, frenet_frame, poly_s, poly_d)
            cartesian_points = samplable_trajectory.sample(time_samples)  # sample cartesian points from the solution

            # validate cartesian points against cartesian limits
            is_valid_in_cartesian = KinematicUtils.filter_by_cartesian_limits(
                cartesian_points[np.newaxis, ...], np.array([0, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED + BIG_EPS]),
                LON_ACC_LIMITS, LAT_ACC_LIMITS)[0]
            if not is_valid_in_cartesian and spec.v <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED + BIG_EPS:
                print("{}. invalid-car action: {}, {}, {}, {}, current velocity: {}".format(
                    behavioral_state.ego_state.timestamp_in_sec, spec.recipe.action_type, spec.recipe.aggressiveness,
                    spec.t, spec.v, iv))
            are_valid.append(is_valid_in_cartesian)

        return are_valid


class FilterForSafetyTowardsTargetVehicle(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """ This is a temporary filter that replaces a more comprehensive test suite for safety w.r.t the target vehicle
         of a dynamic action or towards a leading vehicle in a static action. The condition under inspection is of
         maintaining the required safety-headway + constant safety-margin"""
        # Extract the grid cell relevant for that action (for static actions it takes the front cell's actor,
        # so this filter is actually applied to static actions as well). Then query the cell for the target vehicle
        relative_cells = [(spec.recipe.relative_lane,
                           spec.recipe.relative_lon if isinstance(spec.recipe, DynamicActionRecipe) else RelativeLongitudinalPosition.FRONT)
                          for spec in action_specs]
        target_vehicles = [behavioral_state.road_occupancy_grid[cell][0]
                           if len(behavioral_state.road_occupancy_grid[cell]) > 0 else None
                           for cell in relative_cells]

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA+1)], terminal_fstates[:, :(FS_SA+1)]), axis=1)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        T = np.array([spec.t for spec in action_specs])
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T))

        # solve for s(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)

        are_valid = []
        for poly_s, t, cell, target, spec in zip(poly_coefs_s, T, relative_cells, target_vehicles, action_specs):
            if target is None:
                are_valid.append(True)
                continue
            target_fstate = behavioral_state.extended_lane_frames[cell[LAT_CELL]].convert_from_segment_state(
                target.dynamic_object.map_state.lane_fstate, target.dynamic_object.map_state.lane_id)
            target_poly_s, _ = KinematicUtils.create_linear_profile_polynomials(target_fstate)

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # validate distance keeping (on frenet longitudinal axis)
            is_safe = KinematicUtils.is_maintaining_distance(poly_s, target_poly_s, margin, SAFETY_HEADWAY, np.array([0, t]))
            are_valid.append(is_safe)
        if not np.any(np.array(are_valid)):
            print("{} nothing is safe".format(behavioral_state.ego_state.timestamp_in_sec,))
        # TODO: remove - for debug only
        had_dynmiacs = sum([isinstance(spec.recipe, DynamicActionRecipe) for spec in action_specs]) > 0
        valid_dynamics = sum([valid and isinstance(spec.recipe, DynamicActionRecipe) for spec, valid in zip(action_specs, are_valid)])

        return are_valid
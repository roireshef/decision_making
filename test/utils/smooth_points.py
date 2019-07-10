
import numpy as np
import os
from decision_making.src.global_constants import LAT_ACC_LIMITS, SPLINE_POINT_DEVIATION
from decision_making.src.planning.types import CartesianPath2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from matplotlib import pyplot as p
from typing import List, Dict, Union


class SmoothMapPoints:
    """
    This class smooth lane segments points by preserving continuity and differentiability between downstream lane
    segments, even in case of lanes split & merge.
    """

    @staticmethod
    def smooth_points(path: str, desired_velocities: Union[float, Dict[int, float]],
                      split_lane_id: int = None, merge_lane_id: int = None):
        """
        Given points of downstream lane segments, smooth the points such that the planning will be able to go with
        given desired velocity.
        If 'first_split_lane' and 'last_split_lane' are given, then right/left split points are located between these
        lanes (including). The rest of the points belong to the main lane.
        :param path: directory containing downstream list of lane segments. Each file contains points of one lane segment.
            The points have 3 columns: x (cartesian), y (cartesian), k (curvature).
            File name is <lane_id>.npy. The assumption is that the lane_ids are sorted in the downstream order.
        :param desired_velocities: either dictionary from lane_id to desired velocity or single velocity for all lanes
        :param split_lane_id: first lane_id after the split (optional)
        :param merge_lane_id: last lane_id before the merge (optional)
        """
        # Read from files all points and curvatures. The split points are located between first_split_lane and
        # last_split_lane (including). The rest of the points belong to the main lane.
        all_orig_points, original_k, seams, lane_ids = SmoothMapPoints.read_points_from_files(path=path)

        # if split or merge lanes are given, calculate split/merge points and relevant lane_ids
        split_lane_idx = np.where(lane_ids == split_lane_id)[0][0] if split_lane_id is not None else 0
        merge_lane_idx = np.where(lane_ids == merge_lane_id)[0][0] if merge_lane_id is not None else len(lane_ids) - 1
        relevant_lane_ids = lane_ids[split_lane_idx:merge_lane_idx+1]
        split_point_idx = seams[split_lane_idx - 1] if split_lane_idx > 0 else 0
        merge_point_idx = seams[merge_lane_idx]
        relevant_points = all_orig_points[split_point_idx:merge_point_idx+1]

        # To preserve Frenet frames continuity and differentiability between the split lane and its upstream lane
        # (or merge with its downstream) segment belonging to the main Frenet frame, we perform spline fitting with
        # small overlap between these two segments.
        anchor_size = 10  # in points
        # last prefix point coincides with first curve point, first suffix point coincides with last curve point
        prefix_anchor = all_orig_points[split_point_idx - anchor_size:split_point_idx+1] if split_lane_id is not None else None
        suffix_anchor = all_orig_points[merge_point_idx:merge_point_idx + anchor_size] if merge_lane_id is not None else None

        # fit points by splines; adjust points deviation according to the desired velocity
        velocities_list = [desired_velocities[lane_id] for lane_id in relevant_lane_ids] \
            if type(desired_velocities) is dict else [desired_velocities] * len(relevant_lane_ids)
        frenet, frenet_seams = SmoothMapPoints.fit_points_according_to_velocity(
            relevant_points, prefix_anchor, suffix_anchor, seams[split_lane_idx:merge_lane_idx+1] - split_point_idx,
            velocities_list)

        # save the smoothed points in files: divide all Frenet points to lane segments according to the original seams
        split_orig_k = original_k[split_point_idx:merge_point_idx]
        SmoothMapPoints.save_points_to_files(path + 'smooth/', relevant_lane_ids, frenet_seams, frenet)
        # draw graphs
        SmoothMapPoints.draw_graphs(relevant_points, split_orig_k, frenet, prefix_anchor, suffix_anchor)

    @staticmethod
    def project_points_on_frenet(cpoints: CartesianPath2D, frenet: FrenetSerret2DFrame) -> np.array:
        """
        Convert a large array of cartesian points to Frenet points. Perform the conversion by parts, since memory
        consumption is proportional to points_num^2.
        :param cpoints: array of 2D cartesian points
        :param frenet: Frenet frame
        :return: array of converted Frenet points
        """
        conversion_block_size = 100
        fpoints = np.empty((0, 2))
        for i in range(0, cpoints.shape[0], conversion_block_size):
            fpoints = np.concatenate((fpoints, frenet.cpoints_to_fpoints(cpoints[i:i + conversion_block_size])), axis=0)
        return fpoints

    @staticmethod
    def read_points_from_files(path: str) -> [np.array, np.array, np.array, np.array]:
        """
        Read from files all points and curvatures.
        :param path: input directory
        :return: all points, curvatures, indices of split and merge points, seams between lane segments, lane_ids
        """
        # read file names (lane ids)
        file_names = [f for f in os.listdir(path) if os.path.isfile(path + f)]
        file_names.sort()  # The assumption is that the lane_ids are sorted in the downstream order

        # Read from files all points and curvatures.
        # The split points are located between first_split_lane and last_split_lane (including). The rest of the points
        # belong to the main lane.
        all_points = np.empty((0, 2))
        original_k = np.empty(0)
        seams = []
        lane_ids = []
        for file_name in file_names:
            lane_ids.append(int(file_name.split('.')[0]))

            # load the points
            full_lane_points = np.load(path + file_name)
            lane_points = full_lane_points[:, :2]

            all_points = np.concatenate((all_points, lane_points), axis=0)
            seams.append(all_points.shape[0])

            # read original curvatures
            k = full_lane_points[:, 2]
            original_k = np.concatenate((original_k, k))

        return all_points, original_k, np.array(seams), np.array(lane_ids)

    @staticmethod
    def fit_points_according_to_velocity(points: CartesianPath2D, prefix_anchor: CartesianPath2D, suffix_anchor: CartesianPath2D,
                                         seams: np.array, desired_velocities: List[float]) -> \
            [FrenetSerret2DFrame, np.array]:
        """
        Fit points by splines. The parameter of fitting deviation is determined such that the resulting curvature
        enables to move by given desired velocity.
        The fitting may be anchored in split/merge point to the upstream/downstream main lane points nearby the
        split/merge to preserve continuity and differentiability.
        :param points: 2D array of original cartesian points, including split points and parts of the main lane:
            before the split and after the merge
        :param prefix_anchor: 2D array of points. Normally, prefix_anchor contains a suffix of the upstream lane points.
            It's intended to reach full continuity between the upstream lane segment and this lane.
            Last prefix point should coincide with first curve point.
        :param suffix_anchor: 2D array of points. Normally, suffix_anchor contains a prefix of the downstream lane
            points. It's intended to reach full continuity between this lane and the downstream lane segment.
            First suffix point should coincide with last curve point.
        :param seams: array of indices of points, where lane segments start
        :param desired_velocities: [m/sec] list of desired velocities of the given lane segments
        :return: Frenet frame, seams in Frenet frame
        """
        # find minimal spline point deviation, such that the resulting curvature will enable desired velocity
        point_deviation = SPLINE_POINT_DEVIATION
        frenet = frenet_seams = None
        valid_curvatures = False

        # start with minimal point_deviation parameter and increase it until we get smooth enough points,
        # such that their curvatures comply with the given desired velocities in all lane segments
        while not valid_curvatures:
            frenet = FrenetSerret2DFrame.fit(points, point_deviation=point_deviation,
                                             prefix_anchor=prefix_anchor, suffix_anchor=suffix_anchor)
            # calculate seams in Frenet points
            frenet_seams = np.array([np.argmin(np.linalg.norm(points[seam] - frenet.points, axis=1))
                                     for seam in seams if 0 <= seam < points.shape[0]])

            # loop on all lane segments and validate the obtained curvatures in frenet frame
            valid_curvatures = True
            prev_seam = 0
            for seam, desired_vel in zip(frenet_seams, desired_velocities):
                max_k = np.max(np.abs(frenet.k[prev_seam:seam, 0]))
                if max_k * desired_vel ** 2 > LAT_ACC_LIMITS[1]:
                    valid_curvatures = False
                    break
                prev_seam = seam

            point_deviation *= 1.2

        return frenet, frenet_seams

    @staticmethod
    def save_points_to_files(path: str, lane_ids: np.array, frenet_seams: np.array, frenet: FrenetSerret2DFrame):
        """
        save points in files: divide all Frenet points to lane segments according to the seams
        :param path: output directory
        :param lane_ids: array of lane ids
        :param frenet_seams: array of indices of points, where lane segments start in the Frenet frame
        :param frenet: Frenet frame (with smoothed points)
        """
        # save the smoothed points in files
        # divide all Frenet points to lane segments according to the original seams
        prev_seam = 0
        for lane_id, seam in zip(lane_ids, frenet_seams):
            smooth_points = frenet.points[prev_seam:seam+1]
            smooth_curvatures = frenet.k[prev_seam:seam+1, 0]
            np.save(path + str(lane_id), np.c_[smooth_points, smooth_curvatures])
            prev_seam = seam

    @staticmethod
    def draw_graphs(orig_points: np.array, original_k: np.array, frenet: FrenetSerret2DFrame,
                    prefix: np.array = None, suffix: np.array = None):
        """
        Draw 3 figures with graphs for:
            1. the original and smoothed points,
            2. lateral deviation of the smoothed points,
            3. original and smoothed curvatures.
        :param orig_points: 2D array of original points
        :param original_k: original curvatures
        :param frenet: Frenet frame with smoothed points
        :param prefix: anchor prefix for split points (optional 2D array of points)
        :param suffix: anchor suffix for split points (optional 2D array of points)
        """
        # draw the original and smoothed points
        p.figure()
        p.scatter(orig_points[:, 0], orig_points[:, 1], s=3, marker='.', linewidths=1)
        p.scatter(frenet.points[:, 0], frenet.points[:, 1], s=3, marker='.', linewidths=1)  # draw smoothed points
        if prefix is not None and suffix is not None:
            p.scatter(np.concatenate((prefix[:, 0], suffix[:, 0])), np.concatenate((prefix[:, 1], suffix[:, 1])), s=3,
                      marker='.', linewidths=1)  # draw prefix & suffix

        # draw lateral deviation of the smoothed points from the original points
        f_orig_points = SmoothMapPoints.project_points_on_frenet(orig_points[1:-1], frenet)
        p.figure()
        p.plot(f_orig_points[:, 1])

        # draw original and smoothed curvatures
        p.figure()
        p.plot(original_k)  # original curvatures at the original points
        p.plot(frenet.get_curvature(f_orig_points[:, 0]))  # new curvatures at the original points
        p.show()


# SmoothMapPoints.smooth_points(path="/home/MZ8CJ6/temp/Clinton/", desired_velocities=14)

SmoothMapPoints.smooth_points(path="/home/MZ8CJ6/temp/Clinton/right/",
                              desired_velocities=13.333,         # m/sec
                              split_lane_id=103296514,    # first lane after the split
                              merge_lane_id=103297538    # last lane before the merge
                              )

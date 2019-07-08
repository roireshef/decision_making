
import numpy as np
import os
from decision_making.src.global_constants import LAT_ACC_LIMITS, SPLINE_POINT_DEVIATION
from decision_making.src.planning.types import CartesianPath2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from matplotlib import pyplot as p
from typing import List


class SmoothMapPoints:
    """
    This class smooth lane segments points by preserving continuity and differentiability between downstream lane
    segments, even in case of lanes split & merge.
    """

    @staticmethod
    def straight_connection(path: str):
        """
        Given points of downstream lane segments (typically main lanes with straight connections), smooth the points
        such that the curvatures will enable to move with a desired velocity.
        Save the smoothed points to files by preserving the original points partition to lane segments.
        :param path: directory containing downstream list of lane segments. Each file contains points of one lane segment.
            The points have 3 columns: x (cartesian), y (cartesian), k (curvature).
            File name is <lane_id>.npy. The assumption is that the lane_ids are sorted in the downstream order.
        """
        # read points from the files
        all_points, original_k, _, _, seams, file_names = SmoothMapPoints.read_points_from_files(path=path)

        # smooth all points together
        frenet = FrenetSerret2DFrame.fit(all_points, point_deviation=0.005)

        # save the smoothed points in files: divide all Frenet points to lane segments according to the original seams
        SmoothMapPoints.save_points_to_files(path + 'smooth/', file_names, seams, all_points, frenet)

        # draw original and smoothed points
        SmoothMapPoints.draw_graphs(all_points, original_k, frenet)

        p.figure()
        p.scatter(all_points[:, 0], all_points[:, 1], s=3, marker='.', linewidths=1)  # draw original points
        p.scatter(frenet.points[:, 0], frenet.points[:, 1], s=3, marker='.', linewidths=1)  # draw smoothed points

        # calculate and draw lateral deviations of the smoothed points by projecting the original points on Frenet frame
        f_orig_points = SmoothMapPoints.project_points_on_frenet(all_points[1:-1], frenet)
        p.figure()
        p.plot(f_orig_points[:, 1])

        # draw original and smoothed curvatures at the original points
        p.figure()
        p.plot(original_k)
        k_in_orig_points = frenet.get_curvature(f_orig_points[:, 0])
        p.plot(k_in_orig_points)
        p.show()

    @staticmethod
    def smooth_lane_split_and_merge(path: str, desired_vel: float, split_lane_id: int = None, merge_lane_id: int = None,
                                    split_extention: int = 0):
        """
        Given points of downstream lane segments, smooth the points such that the planning will be able to go with
        given desired velocity.
        If 'first_split_lane' and 'last_split_lane' are given, then right/left split points are located between these
        lanes (including). The rest of the points belong to the main lane.
        :param path: directory containing downstream list of lane segments. Each file contains points of one lane segment.
            The points have 3 columns: x (cartesian), y (cartesian), k (curvature).
            File name is <lane_id>.npy. The assumption is that the lane_ids are sorted in the downstream order.
        :param desired_vel: desired velocity for smoothing curvatures
        :param split_lane_id: first lane_id after the split (optional)
        :param merge_lane_id: last lane_id before the merge (optional)
        :param split_extention: For better points fitting of split/merge lane with too high curvature, the
            splitting/merging lane may be extended backward/forward on expense of the upstream/downstream main lane.
        :return:
        """
        # Read from files all points and curvatures. The split points are located between first_split_lane and
        # last_split_lane (including). The rest of the points belong to the main lane.
        all_orig_points, original_k, split_point_idx, merge_point_idx, seams, file_names = \
            SmoothMapPoints.read_points_from_files(path=path, first_split_lane=split_lane_id, last_split_lane=merge_lane_id)

        # fit points by splines in the interval between split and merge;
        # to improve the fitting of too high curvature split, extend the interval by split_extention from two sides
        ext_split_point_idx = split_point_idx - split_extention
        ext_merge_point_idx = merge_point_idx + split_extention
        frenet, prefix, suffix = SmoothMapPoints.fit_split_points_to_main_lane(
            all_orig_points, ext_split_point_idx, ext_merge_point_idx, desired_vel)

        split_orig_points = all_orig_points[ext_split_point_idx:ext_merge_point_idx]
        # save the smoothed points in files: divide all Frenet points to lane segments according to the original seams
        SmoothMapPoints.save_points_to_files(path + 'smooth/', file_names, seams, split_orig_points, frenet)

        SmoothMapPoints.draw_graphs(split_orig_points, original_k[ext_split_point_idx:ext_merge_point_idx], frenet, prefix, suffix)

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
    def read_points_from_files(path: str, first_split_lane: int = None, last_split_lane: int = None) -> \
            [np.array, np.array, int, int, List[int], List[str]]:
        """
        Read from files all points and curvatures.
        :param path: input directory
        :param first_split_lane: first lane_id after the split (optional)
        :param last_split_lane: last lane_id before the merge (optional)
        :return: all points, curvatures, indices of split and merge points, seams between lane segments, file names
        """
        # read file names (lane ids)
        file_names = [f for f in os.listdir(path) if os.path.isfile(path + f)]
        file_names.sort()  # The assumption is that the lane_ids are sorted in the downstream order

        # Read from files all points and curvatures.
        # The split points are located between first_split_lane and last_split_lane (including). The rest of the points
        # belong to the main lane.
        all_points = np.empty((0, 2))
        original_k = np.empty(0)
        split_point_idx = 0
        merge_point_idx = np.iinfo(np.int32).max
        seams = []
        for file in file_names:
            # load the points
            full_lane_points = np.load(path + file)
            lane_points = full_lane_points[:, :2]
            # get index of the first split point, including split_extention
            if first_split_lane is not None and str(first_split_lane) in file:
                split_point_idx = len(all_points)

            all_points = np.concatenate((all_points, lane_points), axis=0)
            seams.append(all_points.shape[0])

            # get index of the last split point, including split_extention
            if last_split_lane is not None and str(last_split_lane) in file:
                merge_point_idx = len(all_points)

            # read original curvatures
            k = full_lane_points[:, 2]
            original_k = np.concatenate((original_k, k))

        return all_points, original_k, split_point_idx, merge_point_idx, seams, file_names

    @staticmethod
    def fit_split_points_to_main_lane(points: CartesianPath2D, split_point_idx: int, merge_point_idx: int,
                                      desired_vel: float) -> [FrenetSerret2DFrame, CartesianPath2D, CartesianPath2D]:
        """
        Fit points by splines in the interval between split and merge. The parameter of fitting deviation is determined
        such that the resulting curvature enables to move by given desired velocity.
        The fitting is anchored in split/merge point to the upstream/downstream main lane points nearby the split/merge
        to preserve continuity and differentiability.
        :param points: 2D array of original cartesian points, including split points and parts of the main lane:
            before the split and after the merge
        :param split_point_idx: index of the split point in the points array
        :param merge_point_idx: index of the merge point in the points array
        :param desired_vel: [m/sec] desired velocity that determines the fitting deviation parameter
        :return: Frenet frame, anchor prefix & suffix
        """
        # To preserve Frenet frames continuity and differentiability between the split lane and its upstream lane
        # segment belonging to the main Frenet frame, we perform spline fitting with small overlap between these two
        # segments.
        # Similarly, to preserve Frenet frames continuity and differentiability between the merged lane and its
        # downstream lane segment belonging to the main Frenet frame, we perform spline fitting with small overlap
        # between these two segments.
        anchor_size = 10  # in points

        # add prefix and suffix anchors to preserve continuity of the main lane with split/merge lanes
        prefix = points[split_point_idx - anchor_size:split_point_idx+1]  # last prefix point coincides with first curve point
        suffix = points[merge_point_idx-1:merge_point_idx + anchor_size]  # first suffix point coincides with last curve point

        # find minimal spline point deviation, such that the resulting curvature will enable desired velocity
        point_deviation = SPLINE_POINT_DEVIATION
        max_k = np.inf
        frenet = None
        while max_k * desired_vel**2 > LAT_ACC_LIMITS[1]:
            frenet = FrenetSerret2DFrame.fit(points[split_point_idx:merge_point_idx], point_deviation=point_deviation,
                                             prefix_anchor=prefix, suffix_anchor=suffix)
            max_k = np.max(np.abs(frenet.k))
            point_deviation *= 1.2

        return frenet, prefix, suffix

    @staticmethod
    def save_points_to_files(path: str, files: np.array, seams: np.array, points: np.array, frenet: FrenetSerret2DFrame):
        """
        save points in files: divide all Frenet points to lane segments according to the seams
        :param path: output directory
        :param files: array of file names (array of strings)
        :param seams: array of indices of points, where original lane segments start
        :param points: original points
        :param frenet: Frenet frame (with smoothed points)
        """
        # save the smoothed points in files
        # divide all Frenet points to lane segments according to the original seams
        prev_frenet_seam = 0
        for file, seam in zip(files, seams):
            frenet_seam = np.argmin(np.linalg.norm(points[seam] - frenet.points, axis=1)) \
                if seam < points.shape[0] else np.iinfo(np.int32).max
            smooth_points = frenet.points[prev_frenet_seam:frenet_seam+1]
            smooth_curvatures = frenet.k[prev_frenet_seam:frenet_seam+1, 0]
            np.save(path + file, np.c_[smooth_points, smooth_curvatures])
            prev_frenet_seam = frenet_seam

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


# SmoothMapPoints.straight_connection(path="/home/MZ8CJ6/temp/Clinton/")

SmoothMapPoints.smooth_lane_split_and_merge(path="/home/MZ8CJ6/temp/Clinton/right/",
                                            desired_vel=13.333,         # m/sec
                                            split_lane_id=103296514,    # first lane after the split
                                            merge_lane_id=103297538,    # last lane before the merge
                                            split_extention=20          # in points
                                            )

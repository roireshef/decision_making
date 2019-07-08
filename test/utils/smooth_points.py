
import numpy as np
import os
from decision_making.src.global_constants import LAT_ACC_LIMITS, SPLINE_POINT_DEVIATION
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from matplotlib import pyplot as p


def straight_connection(dir: str):
    """
    Given points of downstream lane segments (typically main lanes with straight connections), smooth the points
    such that the curvatures will enable to move with a desired velocity.
    Save the smoothed points to files by preserving the original points partition to lane segments.
    :param dir: directory containing downstream list of lane segments. Each file contains points of one lane segment.
        The points have 3 columns: x (cartesian), y (cartesian), k (curvature).
        File name is <lane_id>.npy. The assumption is that the lane_ids are sorted in the downstream order.
    """
    # read file names (lane ids)
    files = [f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    files.sort()

    # read points from the files
    all_points = np.empty((0, 2))
    original_k = np.empty(0)
    seams = []
    for file in files:
        full_lane_points = np.load(dir + file)
        lane_points = full_lane_points[:, :2]
        all_points = np.concatenate((all_points, lane_points), axis=0)
        seams.append(all_points.shape[0])

        k = full_lane_points[:, 2]
        original_k = np.concatenate((original_k, k))

    # smooth all points together
    frenet = FrenetSerret2DFrame.fit(all_points)

    # save the smoothed points in files
    # divide all Frenet points to lane segments according to the original seams
    prev_frenet_seam = 0
    for file, seam in zip(files, seams):
        frenet_seam = np.argmin(np.linalg.norm(all_points[seam] - frenet.points, axis=1)) \
            if seam < all_points.shape[0] else np.iinfo(np.int32).max
        smooth_points = frenet.points[prev_frenet_seam:frenet_seam+1]
        smooth_curvatures = frenet.k[prev_frenet_seam:frenet_seam+1, 0]
        np.save(dir + 'smooth/' + file, np.c_[smooth_points, smooth_curvatures])
        prev_frenet_seam = frenet_seam

    # draw original and smoothed points
    p.figure()
    p.scatter(all_points[:, 0], all_points[:, 1], s=3, marker='.', linewidths=1)  # draw original points
    p.scatter(frenet.points[:, 0], frenet.points[:, 1], s=3, marker='.', linewidths=1)  # draw smoothed points

    # calculate and draw lateral deviations of the smoothed points by projecting the original points on Frenet frame
    f_orig_points = np.empty((0, 2))
    bulk = 100
    # convert cpoints to fpoints by parts, since memory consumption is proportional to points_num^2
    for i in range(1, all_points.shape[0]-bulk, bulk):
        f_orig_points = np.concatenate((f_orig_points, frenet.cpoints_to_fpoints(all_points[i:i+bulk])), axis=0)
    p.figure()
    p.plot(f_orig_points[:, 1])

    # draw original and smoothed curvatures at the original points
    p.figure()
    p.plot(original_k)
    k_in_orig_points = frenet.get_curvature(f_orig_points[:, 0])
    p.plot(k_in_orig_points)
    p.show()


def smooth_lane_split_and_merge(dir: str, desired_vel: float, first_split_lane: int = None, last_split_lane: int = None):
    """
    Given points of downstream lane segments, smooth the points such that the planning will be able to go with
    given desired velocity.
    If 'first_split_lane' and 'last_split_lane' are given, then right/left split points are located between these
    lanes (including). The rest of the points belong to the main lane.
    :param dir: directory containing downstream list of lane segments. Each file contains points of one lane segment.
        The points have 3 columns: x (cartesian), y (cartesian), k (curvature).
        File name is <lane_id>.npy. The assumption is that the lane_ids are sorted in the downstream order.
    :param desired_vel: desired velocity for smoothing curvatures
    :param first_split_lane: first lane_id after the split (optional)
    :param last_split_lane: last lane_id before the merge (optional)
    :return:
    """
    # For better points smoothing in split lane with too high curvature, the splitting lane should be extended backward.
    # For better points smoothing in merged lane with too high curvature, the merging lane should be extended forward.
    split_extention = 20  # in points

    # To preserve Frenet frames continuity between the split lane and its upstream lane segment belonging to
    # the main Frenet frame, we perform spline fitting with small overlap between these two segments.
    # Similarly, to preserve Frenet frames continuity between the merged lane and its downstream lane segment
    # belonging to the main Frenet frame, we perform spline fitting with small overlap between these two segments.
    # This overlap should be removed after the fitting from the split/merge lanes.
    anchor_size = 10  # in points

    # read file names (lane ids)
    files = [f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    files.sort()  # The assumption is that the lane_ids are sorted in the downstream order

    # Read from files all main-lane points and curvatures.
    # The split points are located between first_split_lane and last_split_lane (including). The rest of the points
    # belong to the main lane.
    all_orig_points = np.empty((0, 2))
    original_k = np.empty(0)
    first_point_idx = 0
    last_point_idx = np.iinfo(np.int32).max
    for file in files:
        # load the points
        full_lane_points = np.load(dir + file)
        lane_points = full_lane_points[:, :2]
        # get index of the first split point, including split_extention
        if first_split_lane is not None and str(first_split_lane) in file:
            first_point_idx = len(all_orig_points) - split_extention

        all_orig_points = np.concatenate((all_orig_points, lane_points), axis=0)
        # get index of the last split point, including split_extention
        if last_split_lane is not None and str(last_split_lane) in file:
            last_point_idx = len(all_orig_points) + split_extention

        # read original curvatures
        k = full_lane_points[:, 2]
        original_k = np.concatenate((original_k, k))

    # add prefix and suffix anchors to preserve continuity of the main lane with split/merge lanes
    prefix = all_orig_points[first_point_idx - anchor_size:first_point_idx+1]  # Last prefix point coincides with first curve point
    suffix = all_orig_points[last_point_idx-1:last_point_idx + anchor_size]  # First suffix point coincides with last curve point

    orig_points = all_orig_points[first_point_idx:last_point_idx]

    # find minimal spline point deviation, such that the resulting curvature will enable desired velocity
    point_deviation = SPLINE_POINT_DEVIATION
    max_k = np.inf
    frenet = None
    while max_k * desired_vel**2 > LAT_ACC_LIMITS[1]:
        frenet = FrenetSerret2DFrame.fit(orig_points, point_deviation=point_deviation, prefix_anchor=prefix, suffix_anchor=suffix)
        max_k = np.max(np.abs(frenet.k))
        point_deviation *= 1.2

    # draw the original and smoothed points
    p.figure()
    p.scatter(orig_points[:, 0], orig_points[:, 1], s=3, marker='.', linewidths=1)
    p.scatter(frenet.points[:, 0], frenet.points[:, 1], s=3, marker='.', linewidths=1)  # draw smoothed points
    p.scatter(np.concatenate((prefix[:, 0], suffix[:, 0])), np.concatenate((prefix[:, 1], suffix[:, 1])), s=3,
              marker='.', linewidths=1)  # draw prefix & suffix

    # draw lateral deviation of the smoothed points from the original points
    if orig_points.shape[0] < 1000:
        f_orig_points = frenet.cpoints_to_fpoints(orig_points[1:-1])
    else:  # convert cpoints to fpoints by parts, since memory consumption is proportional to points_num^2
        bulk = 100
        f_orig_points = np.empty((0, 2))
        for i in range(1, orig_points.shape[0]-bulk, bulk):
            f_orig_points = np.concatenate((f_orig_points, frenet.cpoints_to_fpoints(orig_points[i:i+bulk])), axis=0)
    p.figure()
    p.plot(f_orig_points[:, 1])

    # draw original and smoothed curvatures
    p.figure()
    p.plot(original_k[first_point_idx:last_point_idx])  # original curvatures at the original points
    p.plot(frenet.get_curvature(f_orig_points[:, 0]))  # new curvatures at the original points
    p.show()


# straight_connection(dir="/home/MZ8CJ6/temp/Clinton/")

smooth_lane_split_and_merge(dir="/home/MZ8CJ6/temp/Clinton/right/",
                            desired_vel=13.333,             # m/sec
                            first_split_lane=103296514,     # first lane after the split
                            last_split_lane=103297538       # last lane before the merge
                            )

import pytest
import numpy as np

from rte.ctm.pythonwrappers.src.FrenetSerret2DFrame import FrenetSerret2DFrame

from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.messages.scene_static_enums import NominalPathPoint
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_pg_no_split
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame as DMFrenetSerret2DFrame


def test_frenetConversion_errors(scene_static_pg_no_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_no_split)

    road_ids = MapUtils.get_road_segment_ids()
    print(road_ids)
    for road_id in road_ids:
        print(road_id, ": ", MapUtils.get_lanes_ids_from_road_segment_id(road_id))

    lane_id = 200
    nominal_points = MapUtils.get_lane_geometry(lane_id).a_nominal_path_points

    points = nominal_points[:, (NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value,
                                NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value)]

    yaw = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
    T = np.c_[np.cos(yaw), np.sin(yaw)]
    N = NumpyUtils.row_wise_normal(T)
    k = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value][:, np.newaxis]
    k_tag = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value][:, np.newaxis]
    ds = np.mean(
    np.diff(nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]))  # TODO: is this necessary?

    frenet = FrenetSerret2DFrame.init_from_components(points=points, T=T, N=N, K=k, k_tag=k_tag, ds=ds)

    # Verify feature request - need ds to be a functiional property of FrenetSerret2DFrame
    assert(frenet.ds)

    np.testing.assert_array_equal(frenet.O, points)

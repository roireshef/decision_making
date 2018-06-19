import matplotlib.pyplot as plt
import numpy as np

from decision_making.src.planning.types import FP_SX, FS_DX, FS_SX, C_X, C_Y
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService
from prediction_research.data.ngsim.analysis.map_tools import draw_map
from decision_making.test.prediction.conftest import PREDICTION_HORIZON
from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.werling_trajectory_generator import \
    WerlingTrajectoryGenerator
from decision_making.test.prediction.conftest import original_state_with_sorrounding_objects

DEBUG_PLOT = False

def test_generateTrajectory_sampleParameters_resultPrecise(werling_trajectory_generator: WerlingTrajectoryGenerator,
                                                           original_state_with_sorrounding_objects: State):
    state = original_state_with_sorrounding_objects
    target_obj = state.dynamic_objects[0]

    map_api = MapService.get_instance()

    road_id = target_obj.map_state.road_id
    road_frenet_frame = MapService.get_instance()._rhs_roads_frenet[road_id]

    obj_init_fstate = target_obj.map_state.road_fstate

    # Set final state to advance 20[m] in lon and the width of a road lane in lat
    obj_final_fstate = np.array(obj_init_fstate)
    obj_final_fstate[FS_DX] = obj_init_fstate[FS_DX] + map_api.get_road(road_id=road_id).lane_width
    obj_final_fstate[FS_SX] = obj_init_fstate[FS_SX] + 20.0

    predicted_maneuver_spec = ManeuverSpec(init_state=obj_init_fstate,
                                           final_state=obj_final_fstate,
                                           T_s=PREDICTION_HORIZON,
                                           T_d=PREDICTION_HORIZON / 2.0)

    samplable_trajectory = werling_trajectory_generator.generate_trajectory(timestamp_in_sec=target_obj.timestamp_in_sec,
                                                                 frenet_frame=road_frenet_frame,
                                                                 predicted_maneuver_spec=predicted_maneuver_spec)

    sampling_times = np.linspace(target_obj.timestamp_in_sec, target_obj.timestamp_in_sec + PREDICTION_HORIZON, 30)
    generated_ftrajectory = samplable_trajectory.sample_frenet(time_points=sampling_times)

    generated_ctrajectory = road_frenet_frame.ftrajectory_to_ctrajectory(ftrajectory=generated_ftrajectory)

    assert np.all(np.isclose(obj_final_fstate, generated_ftrajectory[-1]))

    if DEBUG_PLOT:
        plt.subplot(211)
        plt.plot(generated_ftrajectory[:, FS_SX], generated_ftrajectory[:, FS_DX], '-r')
        plt.title('Trajectory in Frenet coordinates')

        plt.subplot(212)
        draw_map()
        plt.plot(generated_ctrajectory[:, C_X], generated_ctrajectory[:, C_Y], '-r')
        plt.title('Trajectory in Map coordinates')

        plt.show()

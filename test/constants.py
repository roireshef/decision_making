from decision_making.paths import Paths
import numpy as np

LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING = "LCM_PUB_SUB_MOCK"

MAP_SERVICE_ABSOLUTE_PATH = 'mapping.src.service.map_service.MapService.get_instance'

SCENE_STATIC_MODEL_ABSOLUTE_PATH = 'SceneStaticModel.get_instance.get_scene_static'

FILTER_OBJECT_OFF_ROAD_PATH = 'decision_making.src.state.state_module.FILTER_OFF_ROAD_OBJECTS'

TP_MOCK_FIXED_TRAJECTORY_FILENAME = Paths.get_resource_absolute_path_filename(
        'fixed_trajectory_files/trajectory_from_recording_2017_11_08_run2.txt')

SOUTH_POINT = np.array([395.96915908, -156.49441364])
NORTH_POINT = np.array([1114.69444225, 8.05576933])
CUSTOM_POINT = np.array([1077.83290929, -27.00159343])

BP_MOCK_FIXED_SPECS = {
    'ROAD_ID': 20,
    'LANE_NUM': 1,
    'TRIGGER_POINT': CUSTOM_POINT,
    'LOOKAHEAD_DISTANCE': 300,  # [m] from TRIGGER_POINT
    'TARGET_VELOCITY': 30,  # [m/sec]
    'PLANNING_TIME': 20  # [sec]
}

BP_NEGLIGIBLE_DISPOSITION_LON = 10  # longitudinal (ego's heading direction) difference threshold
BP_NEGLIGIBLE_DISPOSITION_LAT = 10  # lateral (ego's side direction) difference threshold

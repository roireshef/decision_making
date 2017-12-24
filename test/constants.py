from decision_making.paths import Paths
from pydoc import locate

LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING = "LCM_PUB_SUB_MOCK"

MAP_SERVICE_ABSOLUTE_PATH = 'mapping.src.service.map_service.MapService.get_instance'

# TP_INITIALIZER = locate('decision_making.src.dm_main.DmInitialization')
TP_INITIALIZER = locate('decision_making.test.dm_main_test.DmMockInitialization')
TP_MOCK_FIXED_TRAJECTORY_FILENAME = Paths.get_resource_absolute_path_filename(
    'fixed_trajectory_files/trajectory_from_recording_2017_11_08_run2.txt')

# BP_INITIALIZER = locate('decision_making.src.dm_main.DmInitialization')
BP_INITIALIZER = locate('decision_making.test.dm_main_test.DmMockInitialization')
BP_MOCK_FIXED_SPECS = {
    'ROAD_ID': 20,
    'LANE_NUM': 1,
    'TARGET_LONGITUDE': 1000,  # [m] from road's start
    'TARGET_VELOCITY': 20,  # [m/sec]
    'PLANNING_TIME': 10  # [sec]
}

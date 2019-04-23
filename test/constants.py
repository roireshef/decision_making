from decision_making.paths import Paths

LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING = "LCM_PUB_SUB_MOCK"

MAP_SERVICE_ABSOLUTE_PATH = 'mapping.src.service.map_service.MapService.get_instance'

TP_MOCK_FIXED_TRAJECTORY_FILENAME = Paths.get_resource_absolute_path_filename(
        'fixed_trajectory_files/trajectory_from_recording_2017_11_08_run2.txt')

BP_NEGLIGIBLE_DISPOSITION_LON = 10  # longitudinal (ego's heading direction) difference threshold
BP_NEGLIGIBLE_DISPOSITION_LAT = 10  # lateral (ego's side direction) difference threshold

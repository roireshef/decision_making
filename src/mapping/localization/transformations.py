import numpy as np

from decision_making.src.mapping.localization.coords_transformer import CoordsTransformer
from rte.ctm.src.ctm_api import ctm_api


MICROARCSECS_IN_DEGREES = 3600000


class Transformations():
    transforms = CoordsTransformer()

    def __init__(self):
        pass

    @staticmethod
    def microarcsecs_to_degrees(angle):
        return angle / MICROARCSECS_IN_DEGREES

    @staticmethod
    def transform_location_derivatives_to_agent_frame(north_val, east_val, down_val):
        return ctm_api.transform_location_derivatives_to_agent_frame(north_val, east_val, down_val)

    @staticmethod
    def recalculate_motion_acceleration(accelerations, orientations):
        """
        Calculates the vehicle frame acceleration given the measured accelerations in each axis and the orientations in the IMU Vehicle Frame
        According to documentation here:
        https://support.oxts.com/hc/en-us/articles/115002859149-OxTS-Reference-Frames-and-ISO8855-Reference-Frames
        :param accelerations: Measured accelerations on x, y, z axes in the IMU Vehicle Frame
        :param orientations: Measured orientations in the IMU Vehicle Frame
        :return: The 1X3 vector of accelerations from motion, without gravity
        """
        return ctm_api.recalculate_motion_acceleration(accelerations, orientations)

    @staticmethod
    def transform_location_to_agent_frame(latitude, longitude, altitude, roll, pitch, yaw):
        return ctm_api.transform_agent_location_to_map_frame(latitude, longitude, altitude, roll, pitch, yaw)

    @staticmethod
    def transform_orientation_to_agent_frame(orientation_x, orientation_y, orientation_z):
        return ctm_api.transform_orientation_to_agent_frame(orientation_x, orientation_y, orientation_z)

    @staticmethod
    def convert_geo_to_utm(coords):
        """
        convert points from earth coordinates to utm
        :param coords: array of points in original earth coordinates (lat/lon)
        :return: Nx2 array with [utm_north, utm_east] pairs for each coordinate set
        """
        if coords is None or type(coords) is not np.ndarray or (len(coords.shape) != 2):
            raise ValueError('Coords must be 2d')
        utm_coords = ctm_api.geo_locations_to_world(coords)
        # must flip second coordinate (y) of all points, because CTM returns NWU, while UTM is NED
        utm_coords[:,1] = np.negative(utm_coords[:,1])
        return utm_coords

    @staticmethod
    def _convert_geo_lla_to_xyz(lat_long: np.ndarray) -> np.ndarray:
        from rte.ctm.src.Geodetic import GeoLocationLLA, convert_geo_LLA_to_XYZ
        return convert_geo_LLA_to_XYZ(
                    GeoLocationLLA(
                        latitude=lat_long[0], longitude=lat_long[1], altitude=0.0)).get_array()

    @staticmethod
    def convert_coordinates_to_frame(coords, layer, frame_origin):
        # type: (np.ndarray, int, list) -> np.ndarray
        """
        convert points from earth coordinates to the model frame
        :param coords: array of points in original earth coordinates (lat/lon)
        :param layer: 0 ground level, 1 bridge, 2 bridge above bridge, etc
        :param frame_origin: The frame origin of the map, all points are calculated w.r.t. this frame
        :return: points in model frame with respect to the starting coordinates
        """
        if coords is None or type(coords) is not np.ndarray or (len(coords.shape) != 2):
            raise ValueError('Coords must be 2d')

        Transformations.transforms.transforms.update_map_origin(
                                                  latitude=frame_origin[0]
                                                , longitude=frame_origin[1])

        # Apply the WGS84->UTM transformation:
        #   (latitude,longitude[,altitude=dummy]) -> (x,y[,z=dummy])
        # to all points in the 'coords' matrix and return the 2D UTM (x,y) part
        coords_xyz = np.apply_along_axis(Transformations._convert_geo_lla_to_xyz, 1, coords)
        return Transformations.transforms.transforms.transform_point_array('world', 'map', coords_xyz)[:,0:2]


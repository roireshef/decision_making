from enum import Enum

EPSILON = 0.001
LARGE_NUM = 1000000

EARTH_RADIUS = 6371.  # in kilometers

ROADS_MAP_TILE_SIZE = 10.  # in meters; used by __render_road
RENDER_ROAD_MARGIN = 3.  # in meters; roads are rendered with margins to roads_map

NARROW_ROAD_WITHOUT_MARGINS = 2.6  # in meters; only wider roads have margins
LAYER_HEIGHT = 7.  # in meters; altitude difference between two adjacent layers

LANE_STRIP_WIDTH = 0.15  # in meters; width of a strip between lanes
LANE_STRIP_LENGTH = 3.  # in meters; length of a strip between lanes
LANE_STRIP_CYCLE = 8.  # in meters; lane strip every 8 meters

CUSTOM_NUM_LANES = 2  # lanes number of custom roads
CUSTOM_LANE_WIDTH = 3.  # in meters; lanes' width in custom roads
CUSTOM_ROAD_ROUND_SQUARE_SIZE = 100.  # in meters; size of the edge of the square
CUSTOM_ROAD_ROUND_SQUARE_STRAIGHT_SIZE = 60.  # in meters; size of the straight part of the edge

OSM_FOOTWAY_ROAD_WIDTH = 1.0
OSM_PEDESTRIAN_ROAD_WIDTH = 3.0
OSM_MOTORWAY_ROAD_WIDTH = 3.6
OSM_MOTORWAY_LINK_ROAD_WIDTH = 3.6
OSM_TRUNK_ROAD_WIDTH = 3.5
OSM_TRUNK_LINK_ROAD_WIDTH = 3.5
OSM_PRIMARY_ROAD_WIDTH = 3.3
OSM_PRIMARY_LINK_ROAD_WIDTH = 3.3
OSM_SECONDARY_ROAD_WIDTH = 3.0
OSM_SECONDARY_LINK_ROAD_WIDTH = 3.0
OSM_TERTIARY_ROAD_WIDTH = 3.0
OSM_TERTIARY_LINK_ROAD_WIDTH = 3.0
OSM_RESIDENTIAL_ROAD_WIDTH = 2.7
OSM_SERVICE_ROAD_WIDTH = 2.5
OSM_STEPS_ROAD_WIDTH = 0.8

class Sidewalk(Enum):
    none = 0
    left = 1
    right = 2
    both = 3

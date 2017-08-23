from enum import Enum

# a small number
EPSILON = 0.001
# a large number
LARGE_NUM = 1000000

# [km] Earth radius
EARTH_RADIUS = 6371.

# [m] one-dimensional size of tile (cell) in the map xy2road_map
ROADS_MAP_TILE_SIZE = 10.  # in meters; used by __render_road

# [m] roads are rendered with margins to roads_map
RENDER_ROAD_MARGIN = 3.

# [m] only wider roads have margins
NARROW_ROAD_WITHOUT_MARGINS = 2.6
# [m] altitude difference between two adjacent layers
LAYER_HEIGHT = 7.

# [m] width of a strip between lanes
LANE_STRIP_WIDTH = 0.15
# [m] length of a strip between lanes
LANE_STRIP_LENGTH = 3.
# [m] lane strip every 8 meters
LANE_STRIP_CYCLE = 8.

# lanes number of custom roads
CUSTOM_NUM_LANES = 2
# [m] lanes' width in custom roads
CUSTOM_LANE_WIDTH = 3.
# [m] size of the edge of the square
CUSTOM_ROAD_ROUND_SQUARE_SIZE = 100.
# [m] size of the straight part of the edge
CUSTOM_ROAD_ROUND_SQUARE_STRAIGHT_SIZE = 60.

# [m] Road_width for different road types in OSM file. These numbers were chosen arbitrarily
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

# sidewalk types in OSM
class Sidewalk(Enum):
    no = 'no'
    none = 'none'
    left = 'left'
    right = 'right'
    both = 'both'

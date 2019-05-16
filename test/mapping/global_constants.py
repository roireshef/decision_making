
### Maps
# Map version
MAP_VERSION = '1_5_2'

# 3 Lanes centered on center lane
MAP_3_LANES_RESOURCE_FILE_NAME = 'TestingGroundMap3Lanes.bin'
# 2 lanes centered on 2 West-most lanes
MAP_2_LANES_WEST_RESOURCE_FILE_NAME = 'TestingGroundMap2LanesWest.bin'
# 2 lanes centered on 2 East-most lanes
MAP_2_LANES_EAST_RESOURCE_FILE_NAME = 'TestingGroundMap2LanesEast.bin'
# The default map file
# 3 Lanes centered on center lane
MAP_3_LANES_OLD_RESOURCE_FILE_NAME = 'TestingGroundMap3LanesOld.bin'
# 3 Lanes centered on center lane, python 2.7 compatible
MAP_3_LANES_OLD_PYTHON_27_COMPATABLE_RESOURCE_FILE_NAME = 'TestingGroundMap3LanesOldPython2.7Compatible.bin'

# Milford Oval (circular) Track, 3 lanes centered on center lane
MILFORD_OVAL = 'OvalMilford.bin'

DEFAULT_MAP_FILE = MAP_3_LANES_OLD_PYTHON_27_COMPATABLE_RESOURCE_FILE_NAME

# Amount of error in fitting points to map curve, smaller means using more spline polynomials for the fit (and smaller
# error). This factor is the maximum mean square error (per point) allowed. For example, 0.0001 mean that the
# max. standard deviation is 1 [cm] so the max. squared standard deviation is 10e-4.
SPLINE_POINT_DEVIATION = 0.0001


### Logging
MAP_MODULE_NAME_FOR_LOGGING = "Map Module"
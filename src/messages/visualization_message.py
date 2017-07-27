from enum import Enum

from src.messages.dds_message import DDSMessage
import numpy as np


class RvizVisualizationMessage(DDSMessage):
    """
    The struct used for communicating visualization data to the visualizer modules.
    """
    def __init__(self, _topic: str, _markers_max: int, _frame_id: str, _marker_type: MarkerType, _scale: np.array,
                 _color_a: float, _color_r: float, _color_g: float, _color_b: float,
                 _markers_positions: list[np.array] = None, _markers_orientations: list[np.array] = None):
        self._topic = _topic
        self._markers_max = _markers_max
        self._frame_id = _frame_id
        self._markers_type = _marker_type
        self._scale = _scale
        self._color_a = _color_a
        self._color_r = _color_r
        self._color_g = _color_g
        self._color_b = _color_b
        self._markers_positions = _markers_positions
        self._markers_orientations = _markers_orientations

    def update_marker_arguments(self, positions: list[np.array], orientations: list[np.array]):
        self._markers_positions = positions
        self._markers_orientations = orientations


class MarkerType(Enum):
    ARROW = 0
    CUBE = 1
    SPHERE = 2
    CYLINDER = 3
    LINE_STRIP = 4
    LINE_LIST = 5
    CUBE_LIST = 6
    SPHERE_LIST = 7
    POINTS = 8
    TEXT_VIEW_FACING = 9
    MESH_RESOURCE = 10
    TRIANGLE_LIST = 11

from enum import Enum

from src.messages.dds_message import DDSMessage
import numpy as np


class RvizVisualizationMessage(DDSMessage):
    """
    The struct used for communicating the behavioral debug data to the debug modules.
    It is defined as abstract class, so every instance of behavioral planner would
    implement its' properties in a way that fits it's architecture
    """

    def __init__(self, topic: str, markers_max: int, frame_id: str, marker_type: MarkerType, scale: np.array,
                 color_a: float, color_r: float, color_g: float, color_b: float,
                 markers_positions: list[np.array] = None, markers_orientations: list[np.array] = None):
        self.topic = topic
        self.markers_max = markers_max
        self.frame_id = frame_id
        self.markers_type = marker_type
        self.scale = scale
        self.color_a = color_a
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.markers_positions = markers_positions
        self.markers_orientations = markers_orientations

    def update_marker_arguments(self, positions: list[np.array], orientations: list[np.array]):
        self.markers_positions = positions
        self.markers_orientations = orientations


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

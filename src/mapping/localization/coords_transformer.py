import numpy as np

from rte.ctm.src import CtmService


class CoordsTransformer:

    transforms = None

    def __init__(self):
        CoordsTransformer.init()

    def get_transform(self, source, target):
        return CoordsTransformer.transforms.get_transform_as_homogenous_transform(source, target)

    def get_rotation(self, source, target):
        (rotation, _) = CoordsTransformer.transforms.get_transform_as_rotation_and_translation(source, target)
        return rotation

    def get_translation(self, source, target):
        (_, translation) = CoordsTransformer.transforms.get_transform_as_rotation_and_translation(source, target)
        return translation

    def transform_orientation(self, source, target, orientation_in_source_coords):
        return CoordsTransformer.transforms.transform_orientation_array(
                    source,
                    target,
                    np.array([[orientation_in_source_coords[0],
                               orientation_in_source_coords[1],
                               orientation_in_source_coords[2]]])
               )

    @staticmethod
    def init():
        if CoordsTransformer.transforms is None:
            CoordsTransformer.transforms = CtmService.get_ctm()


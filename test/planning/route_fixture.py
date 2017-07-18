import numpy as np


class RouteFixture:
    @staticmethod
    def get_route(lng: int=200, k: float=0.05, step: int=40, lat: int=100, offset: float=-50.0):
        def stretch(v):
            min = np.min(v[:, 1])
            max = np.max(v[:, 1])

            for i in range(len(v)):
                v[i, 1] = lat * (v[i, 1] - min) / (max - min) + offset

            return v

        return np.concatenate((
            np.array([[i, offset] for i in range(0, lng, step)]),
            stretch(np.array([[i + lng, 1 / (1 + np.exp(-k * (i - lng / 2)))] for i in range(0, lng, step)])),
            np.array([[i + 2 * lng, lat + offset] for i in range(0, lng, step)])
        ), axis=0)
import numpy as np
import pandas as pd

from decision_making.src.prediction.prediction_interface import PredictionInterface

LANE_WIDTH = 4


def decaying_mean_derivative(location, weights=(9, 3, 1)):
    speed = np.diff(location)
    return (np.dot(speed[-(len(weights)):], weights)) / sum(weights)


class Baseline(PredictionInterface):
    def __init__(self):
        self.data = pd.DataFrame(columns=['time_stamp', 'target_id', 'rel_x', 'rel_y'])

    def load_data(self, input_data):
        assert all((x in input_data for x in self.data)), 'self.data does not contain some  field'
        self.data = input_data

    def process(self, host_vx=0):
        """
        simplest possible prediction
        :param host_vx: host absolute lateral speed
        :return:  "lane" relative to host (left / same / right) in next time frame
        """
        # self.data.sort_values(['target_id', 'time_stamp'], inplace=True)
        self.data.set_index(['target_id', 'time_stamp'], inplace=True, verify_integrity=True)
        self.data.sort_index(inplace=True)

        expected = self.data.groupby('target_id').agg(decaying_mean_derivative)
        expected.rename(columns={"rel_x": "rel_vx", "rel_y": "rel_vy", "rel_vx": "rel_ax", "rel_vy": "rel_ay"},
                        inplace=True)
        expected[['rel_x', 'rel_y']] = self.data[['rel_x', 'rel_y']].groupby('target_id').last()
        expected[['rel_x', 'rel_y']] += expected[['rel_vx', 'rel_vy']].to_numpy()
        expected['_rel_x'] = expected['rel_x'] + (expected['rel_vx'] - host_vx)
        expected['lane'] = np.where(expected['_rel_x'] < 0, 'left', 'right')
        expected.loc[abs(expected['rel_x']) < LANE_WIDTH / 2, 'lane'] = 'same'
        return expected[['lane', 'rel_y']]


if __name__ == '__main__':
    seqA, seqB, seqC = 17, 13, 9
    test_data = pd.DataFrame({
        'target_id': ['A'] * seqA + ['B'] * seqB + ['C'] * seqC,
        'time_stamp': pd.date_range('2019-01-01', periods=seqA, freq='0.25S').append(
            [pd.date_range('2019-01-01 00:00:01', periods=seqB, freq='0.25S'),
             pd.date_range('2019-01-01 00:00:02', periods=seqC, freq='0.25S')]),
        'rel_x': np.concatenate((
            np.linspace(-5.5, -1, num=seqA),
            np.linspace(0, 0.1, num=seqB),
            np.linspace(-0.5, 6, num=seqC))),
        'rel_y': np.concatenate((
            np.linspace(-2, 31, num=seqA),
            np.linspace(20, 68, num=seqB),
            np.linspace(10, 36, num=seqC)))
    })

    bl_prediction = Baseline()
    bl_prediction.load_data(test_data)
    output = bl_prediction.process()
    print('*************************\nExpected lane next frame:\n', output)

import os
import shutil

import numpy as np
import pandas

from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame

csv_file_path = 'oval/2d/'
target_frenet_path = 'oval/frenet_objects/'

if __name__ == '__main__':

    if os.path.exists(target_frenet_path):
        shutil.rmtree(target_frenet_path)

    for csv_file_name in os.listdir('oval/2d/'):

        file_id_str = csv_file_name.split('.csv')[0]
        road_segment, lane_num = divmod(int(file_id_str), 2 ** 16)
        print('Now creating Frenet parameters file for segment:', file_id_str)
        target_dir_path = target_frenet_path + '/' + file_id_str + '/'
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
        df = pandas.read_csv(csv_file_path + csv_file_name)
        points = df.iloc[:, 0:2].values
        frenet_object = FrenetSerret2DFrame.fit(points)

        np.save(target_dir_path + 'O.npy', frenet_object.O)
        np.save(target_dir_path + 'T.npy', frenet_object.T)
        np.save(target_dir_path + 'N.npy', frenet_object.N)
        np.save(target_dir_path + 'k.npy', frenet_object.k)
        np.save(target_dir_path + 'k_tag.npy', frenet_object.k_tag)
        np.save(target_dir_path + 'ds.npy', frenet_object.ds)

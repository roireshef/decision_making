import numpy as np
import matplotlib.pyplot as plt
import math
import os
plt.switch_backend('Qt5Agg')

v_mine = []
a_mine = []
lon_if_error = []
lat_if_error = []


f = open('/home/yzystl/av_code/spav/logs/AV_Log_dm_main.log', 'r')
# f = open('/home/yzystl/av_code/spav/logs/AV_Log_rbcar0.log', 'r')
# f = open('/data/recordings/cdrive/Database/2018_02_19/2018_02_19_14_01_Proving_Grounds_-_Daytime/AV_Log_dm_main_2018-02-19_14-03-48.log', 'r')

while True:
    text = f.readline()
    if not text:
        break
    if '\'trajectory\': {\'array\': [' in text:
        v_mine.append(float(text.split('\'trajectory\': {\'array\': [')[1].split(', ')[9]))
        a_mine.append(float(text.split('\'trajectory\': {\'array\': [')[1].split(', ')[10]))
    elif ('lon_lat_errors: [' in text) and ('trajectory_planning_facade' in text):
        lon_if_error.append(float(text.split('lon_lat_errors: [ ')[1].split()[0]))
        lat_if_error.append(float(text.split('lon_lat_errors: [ ')[1].split()[1]))
    elif 'NoValidTrajectoriesFound' in text:
        v_mine.append(0)
        a_mine.append(0)

v_mine = np.asarray(v_mine)
a_mine = np.asarray(a_mine)
lon_if_error = np.asarray(lon_if_error)
lat_if_error = np.asarray(lat_if_error)

fig = plt.figure()
p1 = fig.add_subplot(411)
plt.title('velocity')
p2 = fig.add_subplot(412)
plt.title('acc')
p3 = fig.add_subplot(413)
plt.title('lon_if_error')
p4 = fig.add_subplot(414)
plt.title('lat_if_error')

p1.plot(v_mine)
p2.plot(a_mine)
p3.plot(lon_if_error)
p4.plot(lat_if_error)
plt.show()
print('kak')

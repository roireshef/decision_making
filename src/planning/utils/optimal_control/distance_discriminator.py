from decision_making.src.planning.utils.optimal_control.quintic_poly_formulas import BinaryReadWrite, s_T_grid, \
    v_T_grid, v_0_grid, a_0_grid, st_limits, EPS
from sklearn.utils.extmath import cartesian
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


is_plot = False

predicate_path = 'fine_predicate_wT_0.10_wJ_2.00.bin'
wT_str, wJ_str = predicate_path.split('.bin')[0].split('_')[3], predicate_path.split('.bin')[0].split('_')[5]

predicate_shape = (int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))
predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)

m = [0]
n = np.arange(st_limits[0], st_limits[1]+EPS, 1)

discriminators = cartesian([m, n])

filter_value = np.full(shape=discriminators.shape[0], fill_value=0)
chosen_discriminator = np.zeros(shape=[v_0_grid.shape[0], a_0_grid.shape[0], 2])

for k, v_0 in enumerate(v_0_grid):
    for m, a_0 in enumerate(a_0_grid):
        test = np.zeros(shape=[s_T_grid.shape[0], v_T_grid.shape[0]])
        filter_value = np.zeros(shape=discriminators.shape[0])
        if is_plot:
            plt.figure(k * a_0_grid.shape[0] + m)
            plt.imshow(predicate[k][m], aspect='auto', interpolation='none', extent=extents(v_T_grid) + extents(s_T_grid),
                       origin='lower')
            plt.xlabel(r'$v_T$')
            plt.ylabel(r'$S_T$')
        for c, comb in enumerate(discriminators):
            for i, s_T in enumerate(s_T_grid):
                for j, v_T in enumerate(v_T_grid):
                    test[i, j] = (s_T >= comb[1])  # For s_T lower bound
            test_result = np.multiply(test, predicate[k][m])
            if (test_result == predicate[k][m]).all():
                filter_value[c] = (test == 0).sum()
        chosen_discriminator[k][m] = discriminators[np.argmax(filter_value)]
        print('comb for v_0=%.2f, a_0=%.2f is: %s' % (v_0, a_0, chosen_discriminator[k][m]))
        if is_plot:
            plt.title('comb for v_0=%.2f, a_0=%.2f is: %s' % (v_0, a_0, chosen_discriminator[k][m]))
            plt.show()

BinaryReadWrite.save(chosen_discriminator, 'distance_discriminator_wT_%s_wJ_%s_%dX%dX%d.bin' % (wT_str, wJ_str, v_0_grid.shape[0], a_0_grid.shape[0], 2))

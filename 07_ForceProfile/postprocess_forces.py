import sys
sys.path.append('../sharpy-analysis-tools/')
import numpy as np
import glob
from batch.sets import Actual

### INPUTS
sharpy_output_folder = './output/'
case_name_pattern = 'simple_HALE_uvlm_uinf*_alpha0200'
out_name = 'simple_HALE_uvlm_alpha0200_u_inf'
output_path = './results/'
###

dataset = Actual(sharpy_output_folder + '/' + case_name_pattern + '*')
dataset.systems = ['aeroelastic']
dataset.load_bulk_cases('forces')

alpha_list = []
forces_list = []
for case in dataset.aeroelastic:
    alpha_list.append(case.parameter_value)
    forces_list.append(case.aero_forces)

order = np.argsort(alpha_list)
alpha = np.array([alpha_list[i_order] for i_order in order])
forces = np.zeros((len(alpha), 3))
for ii, idx in enumerate(order):
    forces[ii] = forces_list[idx][1:4]

np.savetxt(output_path + '/' + out_name + '.txt',
           np.column_stack((alpha, forces)))


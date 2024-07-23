import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pandas as pd

import cvxpy as cp
import numpy as np
from numpy import linalg

import multiprocessing as mp
from multiprocessing import Pool, TimeoutError

import time

############# some physiology constants ##########################
RATIO_C_mstruct = 0.44
PROTEINS_MOLAR_MASS_N_RATIO = 0.151
AMINO_ACIDS_MOLAR_MASS_N_RATIO = 0.135
HEXOSE_MOLAR_MASS_C_RATIO = 0.42
AMINO_ACIDS_MOLAR_MASS_C_RATIO = 0.38
PROTEINS_MOLAR_MASS_C_RATIO = 0.38
N_MOLAR_MASS = 14
C_MOLAR_MASS = 12
MG_TO_G = 0.001 # milligram to gram
#################################################################

# @np.vectorize
def calculate_cost(x_coord_para, y_coord_para, total_C_mass, total_N_mass, dry_mass, protein_aa_ratio=10): # x_coord_para is ratio of C in mstruct, y_coord_para is Nstruct ratio of mstruct
    total_C_mole = total_C_mass*MG_TO_G/C_MOLAR_MASS
    total_N_mole = total_N_mass*MG_TO_G/N_MOLAR_MASS

    total_C_sucrose_coefficient = 1
    total_C_AA_coefficient = N_MOLAR_MASS*AMINO_ACIDS_MOLAR_MASS_C_RATIO/(AMINO_ACIDS_MOLAR_MASS_N_RATIO*C_MOLAR_MASS)+protein_aa_ratio*N_MOLAR_MASS*PROTEINS_MOLAR_MASS_C_RATIO/(PROTEINS_MOLAR_MASS_N_RATIO*C_MOLAR_MASS)
    total_C_mstruct_coefficient = x_coord_para/C_MOLAR_MASS

    total_N_sucrose_coefficient = 0
    total_N_AA_coefficient = 1+protein_aa_ratio
    total_N_mstruct_coefficient = y_coord_para/N_MOLAR_MASS

    dry_mass_C_sucrose_coefficient = C_MOLAR_MASS/HEXOSE_MOLAR_MASS_C_RATIO
    dry_mass_AA_coefficient = N_MOLAR_MASS/AMINO_ACIDS_MOLAR_MASS_N_RATIO + protein_aa_ratio*N_MOLAR_MASS/PROTEINS_MOLAR_MASS_N_RATIO
    dry_mass_mstruct_coefficient = 1

    coefficient_matrix = np.array([[total_C_sucrose_coefficient, total_C_AA_coefficient, total_C_mstruct_coefficient],
                                   [total_N_sucrose_coefficient, total_N_AA_coefficient, total_N_mstruct_coefficient],
                                   [dry_mass_C_sucrose_coefficient, dry_mass_AA_coefficient, dry_mass_mstruct_coefficient]])
                                  
    # [TC_1, TC_2, TC_3]
    # [TN_1, TN_2, TN_3]
    # [DM_1, DM_2, DM_3]
    measured_quantity = np.vstack([total_C_mole, total_N_mole, dry_mass]) 
    
    # [sucrose_1 (mol), sucrose_2, sucrose_3]
    # [AA_1, AA_2, AA_3]
    # [mstruct_1, mstruct_2, mstruct_3]
    x = cp.Variable((3,3), pos=True)
    error = cp.max(cp.abs(coefficient_matrix@x - measured_quantity))#cp.sum_squares(coefficient_matrix@x - measured_quantity)
    #######################################################################
    ## alternative condition 1: theoretical sucrose[g]/DM should be 0.1
    # error2 = cp.sum_squares(x[0,:]*12/0.42/measured_quantity[2,:]-0.1)
    ## alternative condition 2: mstruct/DM should be 0.8
    # error2 = cp.sum_squares(x[2,:]/measured_quantity[2,:]-0.8)
    ###########################
    # obj = cp.Minimize(error+0.02*error2)
    # prob = cp.Problem(obj)
    #########################################################################
    ### add mstruct condition as a constraint. This approach seems will give a better result
    constraints = [x[2,:]/measured_quantity[2,:]<=0.8, x[2,:]/measured_quantity[2,:]>=0.6]
    obj = cp.Minimize(error)
    prob = cp.Problem(obj, constraints)
    ####################################################
    prob.solve(solver='SCS')
    
    return error.value, x.value, x_coord_para, y_coord_para, coefficient_matrix
        
def generate_cn_inputs(total_C_mass, total_N_mass, dry_mass, protein_aa_ratio = 10):
    """
    Assume the C/N constitution as following, and solve for each component.
        C = struct_C + sucrose_C + protein_C + aminoacid_C
        N = struct_N + protein_C + aminoacid_C
    Assume proteins:aminoacid = 10:1
    total_C_mass: [mg]
    total_N_mass: [mg]
    dry_mass: [g]
    
    output: err, (sucrose[mol], aa[mol], mstruct[g]), mstruct_C_ratio, Nstruct_ratio, coefficient_matrix
    """
    min_err = np.inf
    best_x = None
    best_Nstruct_ratio = None
    best_RATIO_C_mstruct = None
    RATIO_C_mstruct_values = np.linspace(0.2, 0.8, 100) #100
    Nstruct_ratio_values = np.linspace(1E-5, 0.01, 100) #500
    # Nstruct_ratio_values = np.linspace(0.004, 0.006, 100, endpoint=True) #500
    RATIO_C_mstruct_grid, Nstruct_ratio_grid = np.meshgrid(RATIO_C_mstruct_values, Nstruct_ratio_values, indexing='ij')
    
    n_proc = mp.cpu_count()
    with mp.Pool(processes=n_proc) as pool:
        proc_results =  [ pool.apply_async(calculate_cost, args=(single_c_para, single_n_para, total_C_mass, total_N_mass, dry_mass)) for c_para, n_para in zip(RATIO_C_mstruct_grid, Nstruct_ratio_grid) for single_c_para, single_n_para in zip(c_para, n_para) ]
        results = [r.get() for r in proc_results]
    
    # convert to array format
    results = list(zip(*results))
    results[0] = np.array(results[0])
    results[1] = np.stack(results[1])
    results[2] = np.array(results[2])
    results[3] = np.array(results[3])
    results[4] = np.stack(results[4])
    return results
	
if __name__ == '__main__':
    start_time = time.time()
    data_20240410 = (np.array([47.76408,66.1154,58.88376]), np.array([4.17144,4.64802,4.39452]) , np.array([0.1092,0.1534,0.1404]))
    data_20240415 = (np.array([50,62.4,57.6]), np.array([2.3,1.5,2.0]) , np.array([0.1242,0.1462,0.1288]))
    data_20240422 = (np.array([56,50.5,11.1]), np.array([4.2,4.0,0.6]), np.array([0.1304,0.1424,0.0212]))
    res = generate_cn_inputs(*data_20240422)
    
    
    print('time cost: {}'.format(time.time()-start_time))
    minimal_err_index = np.unravel_index(res[0].argmin(), res[0].shape)
    print('optimal ratio_C_mstruct: {}, optimal Nstruct: {}, min_error: {}'.format(res[2][minimal_err_index], res[3][minimal_err_index], res[0][minimal_err_index]))

    print('the resulted coefficient matrix')
    print(res[4][minimal_err_index])
    print('the optimized result')
    print('sucrose, AA, mstruct: {}, {}, {}'.format(res[1][minimal_err_index][:,0], res[1][minimal_err_index][:,1], res[1][minimal_err_index][:,2]))
    
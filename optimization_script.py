import numpy as np
from scipy.optimize import minimize
import sys
from scipy import stats
import random
from sklearn.decomposition import PCA
import cmapPy.pandasGEXpress.parse as parse
import pandas as pd
import gzip

# Ensure the right seed is set

seed = int(sys.argv[1])
np.random.seed(seed)




encoded_data = np.load('encoded_data.npy')
encoded_dx = encoded_data[:2000]
encoded_b = encoded_data[2000:]


# Define the objective function
def objective_function(u, A, D, lam):
    Au = A @ u
    term1 = np.linalg.norm((D - Au)**2.)**.5 + lam * np.sum(u)
    return term1

def optimize(A, D, lam, seed):
    np.random.seed(seed)
    initial_u = np.random.rand(len(A[0]))
    D2 = (sum(D**2.))**.5
    constraint = {'type': 'ineq', 'fun': lambda u: np.hstack([1 - u, u])}
    result = minimize(objective_function, initial_u, args=(A, D, lam), constraints=constraint)
    optimal_u = result.x
    print('optimal_u = ', optimal_u)
    Au = A @ optimal_u
    R2 = 1. - sum((D - Au)**2.)**.5 / D2
    magu = np.sum(optimal_u)
    return optimal_u, R2, magu

# Define the function to be parallelized
def process_trial(seed, A, D, lam):
    optimal_u, R2, magu = optimize(A, D, lam, seed=seed)
    return optimal_u, R2, magu

# Initialize lists to store results
data_for, data_rev = [], []
R2f, R2r = [], []
magu_f, magu_r = [], []
data_dir = 'data'


# Lambda values to test
lamtab = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10.]

# Select indices randomly
# indI, indF = np.random.choice(np.arange(len(encoded_dx))), np.random.choice(np.arange(len(encoded_dx)))
idx = seed
for lam in lamtab:
    print(f'Trial for lambda: {lam}')
    # for idx, diff in enumerate(dX):

    # Reverse process
        #D = encoded_dx[indI]
    D = encoded_dx[idx]
    A = encoded_b.T
    seed = np.random.choice(np.arange(736353))
    optimal_u, R2, magu = process_trial(seed, A, D, lam)
    data_rev.append(optimal_u)
    np.save(f'{data_dir}/uopt_rev_{idx}_{lam}_.npy', optimal_u)
    R2r.append(R2)
    np.save(f'{data_dir}/R2r_{idx}_{lam}_.npy', R2r)
    magu_r.append(magu)
    np.save(f'{data_dir}/magu_r_{idx}_{lam}_.npy', magu_r)



    # Forward process
    D = -encoded_dx[idx]
    optimal_u, R2, magu = process_trial(seed, A, D, lam)
    data_for.append(optimal_u)
    np.save(f'{data_dir}/uopt_for_{idx}_{lam}_.npy', optimal_u)
    R2f.append(R2)
    np.save(f'{data_dir}/R2f_{idx}_{lam}_.npy', R2f)
    magu_f.append(magu)
    np.save(f'{data_dir}/magu_f_{idx}_{lam}_.npy', magu_f)

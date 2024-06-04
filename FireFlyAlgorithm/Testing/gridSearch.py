import numpy as np
import pandas as pd

from firefly import FireflyAlgorithm
##finds best parameters for the firefly algorithm
def grid_search(csv_file, param_grid):
    best_params = {}
    best_score = np.inf

    for alpha in param_grid['alpha']:
        for beta0 in param_grid['beta0']:
            for gamma in param_grid['gamma']:
                print(f"Testing params alpha: {alpha}, beta: {beta0}, gamma: {gamma}")
                fa = FireflyAlgorithm(n_fireflies=50, max_iter=100, alpha=alpha, beta0=beta0, gamma=gamma)
                results, accuracy = fa.run(csv_file)
                if accuracy < best_score:
                    best_score = accuracy
                    best_params = {'alpha': alpha, 'beta0': beta0, 'gamma': gamma}
                

    return best_params


def find_params(file_name):
    param_grid = {
        'alpha': [0.1, 0.3, 0.5],
        'beta0': [0.5, 1, 1.5],
        'gamma': [0.1, 0.5, 1]
    }
    best_params= grid_search(file_name, param_grid)
    print("Best Parameters:", best_params)

find_params("4Cluster10D.csv")

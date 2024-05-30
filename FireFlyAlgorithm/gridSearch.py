import numpy as np
import pandas as pd

from firefly import FireflyAlgorithm
##finds best parameters for the firefly algorithm
def grid_search(csv_file, param_grid, true_centers):
    df = pd.read_csv(csv_file)
    classes = df['Class'].unique()
    best_params = {}
    best_centers = {}
    best_score = np.inf

    for alpha in param_grid['alpha']:
        for beta0 in param_grid['beta0']:
            for gamma in param_grid['gamma']:
                print(f"Testing params alpha: {alpha}, beta: {beta0}, gamma: {gamma}")
                fa = FireflyAlgorithm(n_fireflies=50, max_iter=100, alpha=alpha, beta0=beta0, gamma=gamma)
                centers = {}
                total_score = 0
                for cls in classes:
                    points = df[df[df.columns[-1]] == cls][df.columns[:-1]].values
                    center = fa.find_center(points, print_output = False)
                    centers[cls] = center
                    true_center = true_centers[cls]
                    total_score += np.sum(abs(true_center-center))
                    #total_score += fa.objective_function(center, points)
                print(f"Total score: {total_score}")
                if total_score < best_score:
                    best_score = total_score
                    best_params = {'alpha': alpha, 'beta0': beta0, 'gamma': gamma}
                    best_centers = centers
                

    return best_params, best_centers


def find_params(file_name, true_centers):
    param_grid = {
        'alpha': [0.1, 0.3, 0.5],
        'beta0': [0.5, 1, 1.5],
        'gamma': [0.1, 0.5, 1]
    }
    best_params, best_centers = grid_search(file_name, param_grid, true_centers)
    print("Best Parameters:", best_params)
    for cls, center in best_centers.items():
        print(f"Center of class {cls}: {center}")

centers = {'A': [30,30], 
           'B': [30,70], 
           'C': [70,30],
           'D': [70,70]}

find_params("4Cluster2DStable.csv", centers)
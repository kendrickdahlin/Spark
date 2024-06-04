import numpy as np
from numpy.random import default_rng
import pandas as pd


class FireflyAlgorithm:
    def __init__(self, pop_size=20, alpha=1.0, beta0=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.rng = default_rng(seed)
        self.lb = 0
        self.ub = 100
        self.max_evals = 100
        self.points = 0
        

    def objective_function(self, center, points):
        return np.sum(np.linalg.norm(self.points-center, axis =1))

    def find_center(self, points):
        self.points = points
        dim = points.shape[0]
        fireflies = self.rng.uniform(self.lb, self.ub, (self.pop_size, dim))
        intensity = np.apply_along_axis(self.objective_function, 1, fireflies)
        best = np.min(intensity)


        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = self.ub - self.lb
        

        while evaluations <= self.max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.beta0 * np.exp(-self.gamma * r)
                        
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], self.lb, self.ub)
                        intensity[i] = self.objective_function(fireflies[i])
                        evaluations += 1
                        best = min(intensity[i], best)
        return best
    
    def run(self,data, print_output = False):
        df = pd.read_csv(data) 
        feature_columns = df.columns[:-1]
        class_column = df.columns[-1]
    
        classes = df[class_column].unique()

        centroids = {}
        for cls in classes:
            points = df[df[class_column] == cls][feature_columns].values
            center = self.find_center(points)
            print(f"Center of class {cls} : {center}")
            centroids[cls] = center
        return centroids
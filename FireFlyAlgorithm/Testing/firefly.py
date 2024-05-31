import numpy as np
import pandas as pd

class FireflyAlgorithm:
    def __init__(self, n_fireflies, max_iter, alpha=0.3, beta0=1.5, gamma=0.1):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def objective_function(self, center, points):
        return np.sum(np.sum((points - center)**2, axis=1))

    def initialize_fireflies(self, dimensions):
        return np.random.uniform(0, 100, (self.n_fireflies, dimensions))

    def distance(self, firefly1, firefly2):
        return np.linalg.norm(firefly1 - firefly2)

    def move_firefly(self, firefly_i, firefly_j):
        r = self.distance(firefly_i, firefly_j)
        beta = self.beta0 * np.exp(-self.gamma * r**2)
        random_factor = self.alpha * (np.random.rand(firefly_i.shape[0]) - 0.5)
        return firefly_i + beta * (firefly_j - firefly_i) + random_factor

    def find_center(self, points, print_output=False, cls = ""):
        fireflies = self.initialize_fireflies(points.shape[1])
        best_firefly = fireflies[0]
        best_fitness = self.objective_function(best_firefly, points)

        if print_output == True:
            print(f"Finding centroid {cls}")
        for k in range(self.max_iter):
            if print_output == True and k%10 == 0:
                print(f"Iteration: {k}/{self.max_iter}")
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j and self.objective_function(fireflies[j], points) < self.objective_function(fireflies[i], points):
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])
                        fitness = self.objective_function(fireflies[i], points)
                        if fitness < best_fitness:
                            best_firefly = fireflies[i]
                            best_fitness = fitness

        return best_firefly

def run(data, alpha = 0.3, beta0 = 1.5, gamma = 0.1, print_output = False):
    fa = FireflyAlgorithm(n_fireflies=50, max_iter=70, alpha=alpha, beta0=beta0, gamma=gamma) 

    df = pd.read_csv(data) 
    feature_columns = df.columns[:-1]
    class_column = df.columns[-1]
  
    classes = df[class_column].unique()

    centroids = {}
    for cls in classes:
        points = df[df[class_column] == cls][feature_columns].values
        center = fa.find_center(points, print_output, cls)
        print(f"Center of class {cls} : {center}")
        centroids[cls] = center
    return centroids

if __name__ == "__main__":
    run("4Cluster2D.csv", True)

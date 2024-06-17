import numpy as np
import pandas as pd

class FireflyAlgorithm:
    def __init__(self, n_fireflies=50, max_iter=20, alpha=0.3, beta0=1, gamma=0.04):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.lb = 0 
        self.ub = 100
        self.points = []
        self.centroids = {}

    def objective_function(self, x):
        #return np.sum(np.sum((self.points - center)**2, axis=1))
        return np.sum(np.linalg.norm(self.points-x, axis = 1))

    def find_center(self):
        dim = self.points.shape[1]

        #initialize fireflies
        fireflies = np.random.uniform(self.lb, self.ub, (self.n_fireflies, dim))
        fitness = np.apply_along_axis(self.objective_function, 1, fireflies)

        #set arbitrary global best
        best_firefly = fireflies[0]
        best_fitness = fitness[0]
        
       
        for k in range(self.max_iter):
            k_alpha = self.alpha * (1-k/self.max_iter) # decreases alpha over time
            ##PARALLELIZE HERE
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] < fitness[i]:
                        #move firefly
                        r = np.linalg.norm(fireflies[i] - fireflies[j]) #distance
                        beta = self.beta0 * np.exp(-self.gamma * r**2) #attractiveness
                        random_factor = k_alpha * (np.random.rand(dim) - 0.5) #randomness

                        #moves firefly based on equation then clips to keep within given interval
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_factor
                        fireflies[i] = np.clip(fireflies[i], self.lb, self.ub)

                        #update fitness
                        fitness[i] = self.objective_function(fireflies[i])
                        #update new best
                        if fitness[i] < best_fitness:
                            best_firefly = fireflies[i]
                            best_fitness = fitness[i]
        return best_firefly

    #returns string of classification
    def classify(self, row):
        distances = {}
        for key, points in self.centroids.items():
            coord = np.array(row)
            distances[key]= np.linalg.norm(points-coord)
        cls = min(distances, key = distances.get)
        return cls
    
    def run(self,file_name):
        df = pd.read_csv(file_name) 
        #split into training and test data

        df = df.sample(frac=1) # shuffle df
        ratio = 0.8
 
        total_rows = df.shape[0]
        train_size = int(total_rows*ratio)
        
        # Split data into test and train
        train_df = df[0:train_size]
        test_df = df[train_size:]

        #train
        feature_columns = train_df.columns[:-1]
        class_column = train_df.columns[-1]
        classes = train_df[class_column].unique()
        classes.sort()

        for cls in classes: 
            self.points = train_df[train_df[class_column] == cls][feature_columns].values
            center = self.find_center()
            #print(f"Center of class {cls} : {center}")
            self.centroids[cls] = center
        
        #test 
        accuracy = 0
        count = 0
        for index, row in test_df.iterrows():
            cls = self.classify(row[:-1])
            if cls == row[-1]:
                accuracy+=1
            count +=1
        #print("Accuracy: ", accuracy/count)
        return self.centroids, accuracy/count

    
if __name__ == "__main__":
    fa = FireflyAlgorithm()
    fa.run("4Cluster2D.csv")

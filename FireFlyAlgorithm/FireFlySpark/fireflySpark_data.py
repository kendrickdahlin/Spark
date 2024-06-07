import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

##SPLITS PARTICLES INTO PARTITIONS
class FireflyAlgorithm:
    def __init__(self, n_fireflies=56, max_iter=20, alpha=0.3, beta0=1, gamma=0.04):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.lb = 0 
        self.ub = 100
        self.centroids = {}
        self.points = []

    def objective_function(self, x):
        return np.sum(np.linalg.norm(np.subtract(self.points,x), axis = 1))

    def find_center(self, points):
        #clean points data
        self.points = [list(i) for i in list(points)]
        dim = len(self.points[0])
        
        #initialize fireflies
        fireflies = np.random.uniform(self.lb, self.ub, (self.n_fireflies, dim))
        fitness = np.apply_along_axis(self.objective_function, 1, fireflies)
        
        
        #set arbitrary global best
        best_firefly = fireflies[0]
        best_fitness = fitness[0]
        
        for k in range(self.max_iter):
            k_alpha = self.alpha * (1-k/self.max_iter) # decreases alpha over time
           
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    ##Here check broadcast variable
                    if fitness[j] < fitness[i]:
                        #move firefly
                        r = np.linalg.norm(np.subtract(fireflies[i], fireflies[j])) #distance
                        beta = self.beta0 * np.exp(-self.gamma * r**2) #attractiveness
                        random_factor = k_alpha * (np.random.rand(dim) - 0.5) #randomness
                        #moves firefly based on equation 
                        fireflies[i] += beta * (np.subtract(fireflies[j],fireflies[i])) + random_factor
                        fireflies[i] = np.clip(fireflies[i], self.lb, self.ub) # keeps new loc within range

                        #update fitness
                        fitness[i] = self.objective_function(fireflies[i])
                        #update new best
                    
                        if fitness[i] < best_fitness:
                            #update global best
                            best_firefly = fireflies[i]
                            best_fitness = fitness[i]
        return best_firefly
    
    #returns string of classification
    def classify(self, point):
        distances = {}
        for cls, centroid in self.centroids.items():
            distances[cls]= np.linalg.norm(np.subtract(centroid,point))
        cls = min(distances, key = distances.get)
        return cls
    

    def run(self, file_name):
        # Create a SparkSession
        spark = SparkSession.builder \
            .appName("Firefly Algorithm with Spark") \
            .getOrCreate()

        sc = spark.sparkContext
        num_cores = sc.defaultParallelism  #Determine the number of available cores
        self.n_fireflies = max(self.n_fireflies, num_cores) 
        

        # Read the dataset from CSV file into a  DataFrame
        df = pd.read_csv(file_name)
        
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
            points = train_df[train_df[class_column] == cls][feature_columns].values
            points_rdd = sc.parallelize(points)
            center = points_rdd.mapPartitions(lambda points: [self.find_center(points)]).collect()
            #clean appearance
            center = list(map(lambda point: list(point), center))
            
            self.centroids[cls] = [sum(x) / len(center) for x in zip(*center)]
            #print(f"Centroid for class {cls}: {self.centroids[cls]}")
                
    
        #test
        accuracy = 0
        count = 0
        for index, row in test_df.iterrows():
            cls = self.classify(row[:-1].values)
            if cls == row[-1]:
                accuracy +=1
            count +=1
        print("Accuracy: ", accuracy/count)
        # Stop the SparkSession
        spark.stop()
        return self.centroids

if __name__ == "__main__":
    fa = FireflyAlgorithm()
    fa.run("Data/Aggregation.csv")
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

    def objective_function(self, x, points):
        #return np.sum(np.sum((self.points - center)**2, axis=1))
        return np.sum(np.linalg.norm(points-x, axis = 1))

    def find_center(self, points, fireflies):
    
        #initialize fireflies
        
        
        fireflies = list(map(lambda firefly: list(firefly), fireflies))
        self.n_fireflies = len(fireflies)
        dim = len(fireflies[0])
        
        fitness = np.apply_along_axis(self.objective_function, 1, fireflies, points)
        
        
        #set arbitrary global best
        best_firefly = fireflies[0]
        best_fitness = fitness[0]
        
       
        for k in range(self.max_iter):
            k_alpha = self.alpha * (1-k/self.max_iter) # decreases alpha over time
           
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] < fitness[i]:
                   
                        #move firefly
                        r = np.linalg.norm(np.subtract(fireflies[i], fireflies[j])) #distance
              
                        beta = self.beta0 * np.exp(-self.gamma * r**2) #attractiveness
                    
                        random_factor = k_alpha * (np.random.rand(dim) - 0.5) #randomness
                        #moves firefly based on equation then clips to keep within given interval
                        fireflies[i] += beta * (np.subtract(fireflies[j],fireflies[i])) + random_factor
                        fireflies[i] = np.clip(fireflies[i], self.lb, self.ub)

                        #update fitness
                        fitness[i] = self.objective_function(fireflies[i], points)
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
    

    def run(self, file_name):
        # Create a SparkSession
        spark = SparkSession.builder \
            .appName("Firefly Algorithm with Spark") \
            .getOrCreate()

        # Determine the number of available cores
        num_cores = spark.sparkContext.defaultParallelism

        # Read the dataset from CSV file into a Spark DataFrame
        df = spark.read.csv(file_name, header=True, inferSchema=True)
        
        #TODO split into training and test data
        
        
        #TODO replace `dim` with code
        dim = 2

        fireflies = np.random.uniform(self.lb, self.ub, (self.n_fireflies, dim))
       

        #add a label to every firefly
        fireflies_rdd = spark.sparkContext.parallelize(fireflies, numSlices=num_cores)
        class_column = df.columns[-1]
        classes = df.select(class_column).distinct()
        
        
        for cls in classes.collect():
            cls = cls[class_column]
            data = df.filter(df[class_column] == cls).drop(class_column).collect()
            
            points = []
            for row in data:
                points.append(list(row))
            
            center = fireflies_rdd.mapPartitions(lambda fireflies: [self.find_center(points, fireflies)]).collect()
            #clean appearance
            center = list(map(lambda point: list(point), center))
            print (f"Center of class {cls}: {center}")
    
        # Stop the SparkSession
        spark.stop()

if __name__ == "__main__":
    fa = FireflyAlgorithm()
    fa.run("4Cluster2D.csv")
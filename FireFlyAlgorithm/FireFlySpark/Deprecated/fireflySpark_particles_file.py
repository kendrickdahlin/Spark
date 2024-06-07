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
        return np.sum(np.linalg.norm(self.points-x, axis = 1))

    def find_center(self, fireflies):
        global_best = open("global_best", "w+")
        #initialize fireflies
        fireflies = list(map(lambda firefly: list(firefly), fireflies))
        n_fireflies = len(fireflies)
        #for i in range (n_fireflies):
            #print (f"Firefly {i} at: {fireflies[i]}")
        dim = len(fireflies[0])
        
        fitness = np.apply_along_axis(self.objective_function, 1, fireflies)
        
        
        #set arbitrary global best
        best_firefly = fireflies[0]
        best_fitness = fitness[0]
        
        for k in range(self.max_iter):
            k_alpha = self.alpha * (1-k/self.max_iter) # decreases alpha over time
           
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    ##Here check file for new best
                    global_best = open("global_best")
                    lines = global_best.readlines()
#                     print("read lines: ", lines)
                    if len(lines)>0:
                        lines = lines[0].split(',')
                        global_firefly = [float(i) for i in lines[:-1]]
#                         print("global firefly: ", global_firefly)
                        global_fitness = float(lines[-1])
#                         print("global fitness: ", global_fitness)
#                         print("best fitness" , best_fitness)
                        
                        if global_fitness < best_fitness:
                            best_fitness = global_fitness
                            best_firefly = global_firefly
                            fireflies[i] = global_firefly
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
                            lines = global_best.readlines()
                            global_best = open("global_best", "w")
                            
                            #update file
                            lines = ','.join(f"{i}" for i in best_firefly)
                            lines = lines+f',{best_fitness}'
                            global_best.writelines(lines)
        return best_firefly, best_fitness

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

        sc = spark.sparkContext
        num_cores = sc.defaultParallelism  #Determine the number of available cores
        self.n_fireflies = max(self.n_fireflies, num_cores) 
        

        # Read the dataset from CSV file into a Spark DataFrame
        df = spark.read.csv(file_name, header=True, inferSchema=True)
        dim = len(df.columns)-1
        
        #split into training and test data
        train, test = df.randomSplit([0.8, 0.2], seed=12345)

        #initialize fireflies
        fireflies = np.random.uniform(self.lb, self.ub, (self.n_fireflies, dim))
        fireflies_rdd = sc.parallelize(fireflies, numSlices=num_cores)
        class_column = train.columns[-1]
        classes = train.select(class_column).distinct().collect()
        
        
        for cls in classes:
            cls = cls[class_column]
            data = train.filter(df[class_column] == cls).drop(class_column).collect()
            
            points = []
            for row in data:
                points.append(list(row))
            self.points = points
            center = fireflies_rdd.mapPartitions(lambda fireflies: [self.find_center(fireflies)]).collect()
            #clean appearance
            center = list(map(lambda point: list(point), center))
            
            #TODO replace code with collect?
            best_centroid = center[0][0]
            best_fitness = center[0][1]
            
            for centroid, fitness in center:
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_centroid = centroid
            self.centroids[cls] = best_centroid
            print (f"Center of class {cls}: {center}")
    
        #test
        accuracy = 0
        count = 0
        for row in test.collect():
            row = list(row)
            cls = self.classify(row[:-1])
            if cls == row[-1]:
                accuracy +=1
            count +=1
        print("Accuracy: ", accuracy/count)
        # Stop the SparkSession
        spark.stop()

if __name__ == "__main__":
    fa = FireflyAlgorithm()
    fa.run("4Cluster2D.csv")
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import csv

spark = SparkSession.builder.appName("Firefly Algorithm with Spark").getOrCreate()
sc = spark.sparkContext

#define objective function
def objective_function(firefly, X,y):
    pred = predict(firefly, X)
    mse = np.mean(np.subtract(y,pred)**2)
    return mse

#define firefly algorithm
def firefly(X,y, n_fireflies):
    #print message in each node
    print(f"Data: {X}")
    
    #set up params
    n_fireflies = n_fireflies
    max_iter = 100
    gamma = 0.5  
    delta = 0.7 #how much it moves towards best firefly
    lb = -5
    ub = 5
    dim = X.shape[1]+1
    
    #initialize fireflies
    fireflies = np.random.uniform(lb,ub,(n_fireflies,dim))
    fitness = np.apply_along_axis(objective_function, 1, fireflies, X,y)
    
    #set global best
    gbest_firefly = fireflies[np.argmin(fitness)]
    gbest_fitness = np.min(fitness)
    
    for k in range(max_iter):
        for i in range(n_fireflies):
            pbest_attractiveness = 0
            pbest_firefly = fireflies[i]
            for j in range(n_fireflies):
                #if j is better, move i towards it
                if fitness[j] < fitness[i]:
                    r = np.sum(abs(np.subtract(fireflies[j], fireflies[i]))**2) #distance squared
                    beta1 = fitness[j]*np.exp(-gamma * r) #attractiveness
                    if beta1 > pbest_attractiveness:
                        pbest_attractiveness = beta1
                        pbest_firefly = fireflies[j]
            fireflies[i] += delta * np.subtract(pbest_firefly,fireflies[i]) #proportion
            fitness[i] = objective_function(fireflies[i], X, y)

            if fitness[i] < gbest_fitness:
                gbest_fitness = fitness[i]
                gbest_firefly = fireflies[i]
   
    return gbest_firefly


#classifies input
def predict(model, X):
    pred = (np.dot(X,model[:-1])+model[-1]>=0).astype(int)
    return pred

def run(TestNum, FilePath, NumParticles, StartTime, TestList):
    #read data
    df = spark.read.csv(FilePath, header=True, inferSchema=True)
    X = np.array(df.select(df.columns[:-1]).collect())
    y = np.array(df.select(df.columns[-1]).collect()).flatten()
    
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
   
 
    #Create an RDD of (feature, label) pairs
    data_rdd = sc.parallelize(list(zip(X, y)))

    #Firefly algorithm applied to partitions
    def firefly_partition(partition):
        partition_list = list(partition)
        if len(partition_list) == 0:
            return []
        X_partition, y_partition = zip(*partition_list)
        X_partition = np.array(X_partition)
        y_partition = np.array(y_partition)
        return [firefly(X_partition, y_partition, NumParticles)]

    weights = data_rdd.mapPartitions(firefly_partition).collect()
    model = [sum(x) / len(weights) for x in zip(*weights)]

    y_pred = predict(model,X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    
    TestList.append((TestNum, FilePath, NumParticles, accuracy, precision, recall, f1, (time.time()-StartTime)))
        
    return TestList
        
    
def repeat_csv(output_file, repetitions):
    input_file = 'Behavior.csv'
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)    # Read the rest of the rows

    # Write to the output file
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header once
        
        for _ in range(repetitions):
            writer.writerows(rows)  # Write the rows repeatedly
            

TestList= [ ]

FilePath = 'Test.csv'
TestNum = 1
NumParticles = 50 
NumOfFold = 56
StartTime = time.time()

repeat_csv(FilePath,NumOfFold)
run(TestNum,FilePath,NumParticles,StartTime,TestList)
print(TestList)

spark.stop()
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
def firefly(X,y,fireflies):
    #print message in each node
    print(f"Fireflies: {fireflies}")

    #set up params
    max_iter = 100
    gamma = 0.5  
    delta = 0.7 #how much it moves towards best firefly
    
    #initialize fireflies
    fireflies = list(map(lambda firefly: list(firefly), fireflies))
    fitness = np.apply_along_axis(objective_function, 1, fireflies,X,y)
    n_fireflies = len(fireflies)
    
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
    n_fireflies = NumParticles
    lb = -5
    ub = 5

    num_cores = sc.defaultParallelism  #Determine the number of available cores
    n_fireflies = max(n_fireflies, num_cores) 
    

    # Read the dataset from CSV file into a Spark DataFrame
    df = spark.read.csv(FilePath, header=True, inferSchema=True)
    X = np.array(df.select(df.columns[:-1]).collect())
    y = np.array(df.select(df.columns[-1]).collect()).flatten()
    
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #create an RDD of fireflies
    dim = len(df.columns)
    fireflies = np.random.uniform(lb, ub, (n_fireflies, dim))
    fireflies_rdd = sc.parallelize(fireflies)

  
    #firefly algorithm applied to partitions

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



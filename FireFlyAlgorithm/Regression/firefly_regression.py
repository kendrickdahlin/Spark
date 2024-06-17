import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#define objective function
def objective_function(firefly, X,y):
    pred = (np.dot(X,firefly[:-1])+firefly[-1]>=0).astype(int)
    mse = np.mean(np.subtract(y,pred)**2)
    return mse

#define firefly algorithm
def firefly(X,y, delta):
    #set up params
    n_fireflies = 50
    max_iter = 100
    gamma = 0.5  
    delta = delta #how much it moves towards best firefly
    lb = -5
    ub = 5
    dim = X.shape[1]+1
    
    #initialize fireflies
    fireflies = np.random.uniform(lb,ub,(n_fireflies,dim))
    fitness = np.apply_along_axis(objective_function, 1, fireflies, X,y)
    
    gbest_firefly = fireflies[np.argmin(fitness)]
    gbest_fitness = np.min(fitness)
    
    fitness_over_time = [[i] for i in fitness]
    firefly_positions = [fireflies.copy()]
    
    for k in range(max_iter):
        for i in range(n_fireflies):
            pbest_fitness = 0
            pbest_firefly = fireflies[i]
            for j in range(n_fireflies):
                #if j is better, move i towards it
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(np.subtract(fireflies[j], fireflies[i])) #distance
                    beta1 = np.exp(-gamma * r**2) #attractiveness
                    if beta1 > pbest_fitness:
                        pbest_fitness = beta1
                        pbest_firefly = fireflies[j]
                    
            
            fireflies[i] += delta * np.subtract(pbest_firefly,fireflies[i]) #proportion
            fitness[i] = objective_function(fireflies[i], X, y)
            fitness_over_time[i].append(fitness[i])

            if fitness[i] < gbest_fitness:
                gbest_fitness = fitness[i]
                gbest_firefly = fireflies[i]
        firefly_positions.append(fireflies.copy())
     
                
    #GRAPH
    """
    plt.figure(figsize=(10, 6))
    for i in range (n_fireflies):
        plt.plot(fitness_over_time[i], label='Best Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.yscale('log')
    plt.title('Fitness Over Time')
    plt.grid(True)
    plt.show()
    
    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    firefly_positions_2d = [pca.fit_transform(positions) for positions in firefly_positions]

    # Plot firefly positions over time
    plt.figure(figsize=(10, 6))
    for i in range(0,6):
        positions = np.array([firefly_positions_2d[t][i] for t in range(max_iter + 1)])
        plt.plot(positions[:, 0], positions[:, 1], marker='o', label='Firefly {}'.format(i))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Firefly Positions Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
    """

    return gbest_firefly


#classifies input
def predict(model, X):
    pred = (np.dot(X,model[:-1])+model[-1]>=0).astype(int)
    return pred


def run(file_name):
    spark = SparkSession.builder \
            .appName("Firefly Algorithm with Spark") \
            .getOrCreate()
    sc = spark.sparkContext

    #read data
    df = spark.read.csv(file_name, header=True, inferSchema=True)
    X = np.array(df.select(df.columns[:-1]).collect())
    y = np.array(df.select(df.columns[-1]).collect()).flatten()
    
    
    #transform y values to ints
    y = LabelEncoder().fit_transform(y)
    
    #scale X values
    X = StandardScaler().fit_transform(X)

    model = firefly(X,y)
    y_pred = predict(model,X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
        
    
if __name__ == "__main__":
    run("Behavior.csv")
import numpy as np
from pyspark.sql import SparkSession

#define objective function
def objective_function(firefly, X,y):
    pred = predict(firefly, X)
    mse = np.mean(np.subtract(y,pred)**2)
    return mse

#define firefly algorithm
def firefly(X,y):
    #set up params
    n_fireflies = 50
    max_iter = 100
    gamma = 0.5  
    delta = 0.7 #how much it moves towards best firefly
    lb = -5
    ub = 5
    dim = X.shape[1]+1
    
    #initialize fireflies
    fireflies = np.random.uniform(lb,ub,(n_fireflies,dim))
    fitness = np.apply_along_axis(objective_function, 1, fireflies, X,y)
    
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

#manual label encoding
def label_encode(y):
    classes = np.unique(y)
    class_to_index = {c: idx for idx, c in enumerate(classes)}
    y_encoded = np.array([class_to_index[label] for label in y])
    return y_encoded

#manual standardization
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled

# Manual accuracy calculation
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

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
    y= label_encode(y)
    
    #scale X values
    X = standardize(X)

    model = firefly(X,y)
    y_pred = predict(model,X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    spark.stop()
    
if __name__ == "__main__":
    run("Behavior.csv")
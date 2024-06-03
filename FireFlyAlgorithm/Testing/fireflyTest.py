##testing firefly algorithm with multiple different datasets

import numpy as np
import pandas as pd
import firefly as ff
import generateData as gd4C2D
import generate2Cluster4D as gd2C4D
from generateData import generate_data

# create an instance of the firefly class
fa = ff.FireflyAlgorithm()

clusters = 4
dimensions = 2

file_name = f"{clusters}Cluster{dimensions}D.csv"

#generate training data
ranges = generate_data(5000, clusters, dimensions)
#print("Ranges: ", ranges)

#run the model
result = fa.run(file_name)

#generate testing data 
generate_data(100,clusters, dimensions)

#test
print("TESTING")
df = pd.read_csv(file_name) 

def classify(row):
    distances = {}
    for key, points in result.items():
        coord = np.array(row[:-1])
        distances[key]= np.linalg.norm(points-coord)
    cls = min(distances, key = distances.get)
    if cls == row[-1]:
       return True
    return False

accuracy = 0
count = 0
for index, row in df.iterrows():
    if classify(row):
        accuracy+=1
    count +=1
print("Accuracy: ", accuracy/count)



    







    
  
    



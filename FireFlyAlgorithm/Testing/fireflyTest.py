##testing firefly algorithm with multiple different datasets

import numpy as np
import pandas as pd
import firefly as ff
import generateData as gd4C2D
import generate2Cluster4D as gd2C4D
from generateData import generate_data

# create an instance of the firefly class
fa = ff.FireflyAlgorithm(gamma=0.3)

clusters = 4
dimensions = 2
size = 50

file_name = f"{clusters}Cluster{dimensions}D.csv"

#generate training data
ranges = generate_data(size, clusters, dimensions)
#print("Ranges of data: ", ranges)

#run the model
result, accuracy = fa.run(file_name)
print(result)
print(accuracy)
 







    
  
    



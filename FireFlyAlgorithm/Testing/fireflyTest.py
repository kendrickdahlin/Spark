##testing firefly algorithm with multiple different datasets

import firefly as ff
import generateData as gd4C2D
import generate2Cluster4D as gd2C4D
from generateData import generate_data

#fa = FireflyAlgorithm(n_fireflies=50, max_iter=100)

clusters = 2
dimensions = 4


fileName = f"{clusters}Cluster{dimensions}D.csv"

true_centers = generate_data(10, clusters, dimensions)
print("True Centers: ", true_centers)

result = ff.run(fileName)

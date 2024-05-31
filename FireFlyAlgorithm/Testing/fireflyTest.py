##testing firefly algorithm with multiple different datasets

import firefly as ff
import generateData as gd4C2D
import generate2Cluster4D as gd2C4D
from generateData import generate_data

#fa = FireflyAlgorithm(n_fireflies=50, max_iter=100)

clusters = 2
dimensions = 10


fileName = f"{clusters}Cluster{dimensions}D.csv"

#ranges = generate_data(5000, clusters, dimensions)
#print("Ranges: ", ranges)

alpha = 0.3
beta0 = 1
gamma = 1
result = ff.run(fileName, alpha, beta0, gamma)

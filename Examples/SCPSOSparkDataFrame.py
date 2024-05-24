import numpy as np
import time
from pyspark.sql import SparkSession # type: ignore
from pyspark.sql.functions import col # type: ignore

Spark = SparkSession.builder.appName('SCPSO').getOrCreate()

start_time = time.time()

# Read the dataset from the CSV file
data = Spark.read.csv('ScpsoDataset.csv', header=True, inferSchema=True)
data.show()  # Displaying the DataFrame

"""
# PSO parameters
num_particles = 20
max_iterations = 10
inertia_weight = 0.7
cognitive_coefficient = 1.5
social_coefficient = 1.5
num_classes = data.select('label').distinct().count()  # Number of unique classes

# Initialize particles with random centroids
def initialize_particles():
    particles = []
    for _ in range(num_particles):
        centroids = np.random.rand(num_classes, 2) * 100  # Random initial centroids, scaled to match data range
        velocity = np.zeros((num_classes, 2))  # Initialize velocity to zeros
        particles.append({'centroids': centroids, 'velocity': velocity, 'pbest_centroids': centroids.copy(), 'pbest_fitness': np.inf})
    return particles

# Fitness function F1
def fitness_F1(centroids):
    distances = []
    for row in data.collect():
        x = row['x']
        y = row['y']
        label = row['label']
        centroid = centroids[ord(label) - ord('A')]  # Convert label to index
        distances.append(np.linalg.norm([x - centroid[0], y - centroid[1]]))
    return np.mean(distances)

# Fitness function Fpsi
def fitness_Fpsi(centroids):
    incorrect_count = data.rdd.map(lambda row: (row['x'], row['y'], row['label'])).map(lambda row: (row[0], row[1], row[2], np.argmin(np.linalg.norm(centroids - [row[0], row[1]], axis=1)))).filter(lambda x: x[3] != ord(x[2]) - ord('A')).count()
    return incorrect_count / data.count()  # Percentage of incorrectly classified instances

# Fitness function F2
def fitness_F2(centroids):
    f1 = fitness_F1(centroids)
    fpsi = fitness_Fpsi(centroids)
    return (f1 + fpsi) / 2

# PSO update equations
def update_velocity_position(particle, gbest_centroids):
    global inertia_weight, cognitive_coefficient, social_coefficient
    r1 = np.random.rand()
    r2 = np.random.rand()
    particle['velocity'] = (inertia_weight * particle['velocity'] +
                            cognitive_coefficient * r1 * (particle['pbest_centroids'] - particle['centroids']) +
                            social_coefficient * r2 * (gbest_centroids - particle['centroids']))
    particle['centroids'] += particle['velocity']

# Main PSO function
def pso():
    particles = initialize_particles()
    gbest_centroids = None
    gbest_fitness = np.inf

    for iteration in range(max_iterations):
        for particle in particles:
            fitness = fitness_F2(particle['centroids'])  # Use F2 as the fitness function
            if fitness < particle['pbest_fitness']:
                particle['pbest_fitness'] = fitness
                particle['pbest_centroids'] = particle['centroids'].copy()
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_centroids = particle['centroids'].copy()
        for particle in particles:
            update_velocity_position(particle, gbest_centroids)

    return gbest_centroids

# Example usage

best_centroids = pso()
print("Best centroids found by PSO:")
print(best_centroids)
end_time = time.time()

# Step 3: Calculate the duration
duration = end_time - start_time

# Step 4: Print the duration
print("Execution time:", duration, "seconds")

Spark.stop()
"""

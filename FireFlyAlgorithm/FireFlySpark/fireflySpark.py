from pyspark.sql import SparkSession
import numpy as np

# Define the FireflyAlgorithm class
class FireflyAlgorithm:
    def __init__(self, n_fireflies, max_iter, alpha=0.3, beta0=1, gamma=0.2, convergence_threshold=1e-6, patience=10):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold
        self.patience = patience

    def objective_function(self, center, points):
        return np.sum(np.sum((points - center)**2, axis=1))

    def initialize_fireflies(self, dimensions):
        return np.random.uniform(0, 100, (self.n_fireflies, dimensions))

    def distance(self, firefly1, firefly2):
        return np.linalg.norm(firefly1 - firefly2)

    def move_firefly(self, firefly_i, firefly_j, iteration):
        r = self.distance(firefly_i, firefly_j)
        beta = self.beta0 * np.exp(-self.gamma * r**2)
        alpha = self.alpha * (1 - iteration / self.max_iter)  # decreases alpha over time
        random_factor = alpha * (np.random.rand(firefly_i.shape[0]) - 0.5)
        return firefly_i + beta * (firefly_j - firefly_i) + random_factor

    def find_center(self, points, print_output=False, cls=""):
        fireflies = self.initialize_fireflies(points.shape[1])
        best_firefly = fireflies[0]
        best_fitness = self.objective_function(best_firefly, points)
        no_improvement_count = 0

        if print_output:
            print(f"Finding centroid {cls}")
        for k in range(self.max_iter):
            if print_output and k % 10 == 0:
                print(f"Iteration: {k}/{self.max_iter}")

            previous_best_fitness = best_fitness

            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j and self.objective_function(fireflies[j], points) < self.objective_function(fireflies[i], points):
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], k)
                        fitness = self.objective_function(fireflies[i], points)
                        if fitness < best_fitness:
                            best_firefly = fireflies[i]
                            best_fitness = fitness

            if abs(previous_best_fitness - best_fitness) < self.convergence_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if no_improvement_count >= self.patience:
                print(f"Convergence reached at iteration {k}")
                break

        return best_firefly

# Initialize SparkSession
spark = SparkSession.builder.appName("FireflyAlgorithmSpark").getOrCreate()

# Load data into a Spark DataFrame
file_path = '4Cluster2D.csv'
spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Extract feature columns and class column
columns = spark_df.columns
feature_columns = columns[:-1]
class_column = columns[-1]

# Define a function to apply the Firefly Algorithm to each group
def process_partition(partition):
    fa = FireflyAlgorithm(n_fireflies=50, max_iter=70, alpha=0.3, beta0=1.5, gamma=0.1)
    results = []
    data = list(partition)
    if data:
        classes = set(row[class_column] for row in data)
        for cls in classes:
            points = np.array([tuple(row[col] for col in feature_columns) for row in data if row[class_column] == cls])
            center = fa.find_center(points, print_output=False, cls=cls)
            results.append((cls, center.tolist()))
    return results

# Convert Spark DataFrame to RDD and apply the Firefly Algorithm to each group using mapPartitions
rdd = spark_df.rdd.mapPartitions(lambda partition: process_partition(partition))

# Collect and display the results
results = rdd.collect()
for cls, center in results:
    print(f"Center of class {cls}: {center}")

# Stop the SparkSession
spark.stop()

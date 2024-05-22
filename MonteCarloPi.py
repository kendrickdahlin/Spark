#Author: Aaron Mckenzie
# This code randomly samples points and calculates how many fall within a unit circle, then estimates π based on the ratio of points within the circle to the total number of points.

from pyspark.sql import SparkSession
import random

spark = SparkSession.builder.appName("Pi Calculation").getOrCreate()

def estimate_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1

    pi_estimate = 4.0 * inside_circle / num_samples
    return pi_estimate

num_samples = 10000000
pi_estimate = estimate_pi(num_samples)
print("Estimated value of Pi is:", pi_estimate)
spark.stop()

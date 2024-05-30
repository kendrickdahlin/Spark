#Implements the mergeSort algorithm using


import random
from pyspark.sql import SparkSession
spark = SparkSession.builder\
    .appName("MergeSort")\
    .getOrCreate()

sc = spark.sparkContext
def foo(arr):
    return map(lambda x: x*x, arr)
#arr = [random.randint(0,10) for _ in range(20)]
df = spark.read.option("header", True).csv("Data.csv")

rdd = sc.parallelize(df.collect())

#result = rdd.map(lambda x: [mergeSort(list(x))])
result = rdd.mapPartitions(lambda x: foo(list(x))).collect()




spark.stop()



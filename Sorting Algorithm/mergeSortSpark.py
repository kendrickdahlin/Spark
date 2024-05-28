#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from pyspark.sql import SparkSession
spark = SparkSession.builder\
    .appName("MergeSort")\
    .getOrCreate()

sc = spark.sparkContext

def mergeSort(list):
    list_length = len(list)

    if list_length == 1:
        return list

    q = list_length // 2 #calculate the central point
    left = mergeSort(list[:q])
    right = mergeSort(list[q:])

    return merge(left, right)

def merge(left, right):
    ordered = []
    while left and right:
        ordered.append((left if left[0] <= right[0] else right).pop(0))
    return ordered + left + right

#arr = [random.randint(0,10) for _ in range(20)]

def sparkSort(list):
    print("")
    print("====Initital List======")
    print(list)
    print("")

    if len(list) == 1:
        return list
    q = len(list) // 2 #calculate the central point
    left = mergeSort(list[:q])
    right = mergeSort(list[q:])

    return merge(left, right)


df = spark.read.option("header", True).csv("Data.csv")

rdd = sc.parallelize(df.collect())

rdd.map(lambda x: print(x))

#result = rdd.map(lambda x: [mergeSort(list(x))])
result = rdd.mapPartitions(lambda x: [sparkSort(list(x))])

partial_sorted = result.collect()
print("")
print("====Partial Sort======")
print(partial_sorted)
print("")
print("")

sorted_array = result.reduce(merge)

print("")
print("====Sorted Array======")
print(sorted_array)
print("")
print("")


spark.stop()



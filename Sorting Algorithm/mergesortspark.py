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

def sparkSort(df):
    #print out initial array
    # convert array to RDD
    rdd = df.rdd

    #sort each partition independently using mergesort and print the sorted array within each partition
    sorted_rdd = rdd.mapPartitions(lambda partition: [mergeSort(list(partition))])

    # collected sorted elements from all partitions
    sorted_arr = sorted_rdd.flatMap(lambda x: x).collect()

    #print sorted array within each partition
    partition_sorted_arr = sorted_rdd.collect()
    

    print("")
    print("")
    print("")
    print("Partially sorted arrays: ")
    print("")

    for i, partition in enumerate(partition_sorted_arr):
        print("Partition", i, "Sorted Array: ", partition[0], "...", partition[-1])
        print("")
    finalSortedArr = sorted(sorted_arr)
    print("")
    print("Final sorted array:")
    print("---------")
    print("")
    print(finalSortedArr[0], "...", finalSortedArr[-1])
    print("")
    print("")
    print("")
    return finalSortedArr
# Main function


arr = [random.randint(0,1000) for _ in range(10000)]

#df = spark.createDataFrame([(x,) for x in arr])
df = spark.read.option("header", True).csv("Data.csv")

sortedArray = sparkSort(df)

spark.stop()
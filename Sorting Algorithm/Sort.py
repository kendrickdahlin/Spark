from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("ArrayArrange") \
    .getOrCreate()

# Get the SparkContext from the SparkSession
sc = spark.sparkContext

# Define the quicksort function
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        lesser = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quicksort(lesser) + [pivot] + quicksort(greater)

# Define the function to arrange the array using quicksort within partitions
def arrange_array(arr):
    # Convert the array to an RDD
    rdd = arr.rdd
    
    # Sort each partition independently using quicksort and print the sorted array within each partition
    sorted_rdd = rdd.mapPartitions(lambda partition: [quicksort(list(partition))])

    # Collect the sorted elements from all partitions
    sorted_arr = sorted_rdd.flatMap(lambda x: x).collect()

    # Print the sorted array within each partition
    partition_sorted_arr = sorted_rdd.collect()
    for i, partition_arr in enumerate(partition_sorted_arr):
        print("Partition", i, "Sorted Array:", partition_arr)

    final_sorted_arr = quicksort(sorted_arr)
    
    # Print the final sorted array without row and value information
    print("Final Sorted Array:")
    print(final_sorted_arr)

    return final_sorted_arr

# Main function
if __name__ == "__main__":
    # Sample array of integers
    array = [74, 210, 35, 127, 185, 392, 224, 321, 129, 222, 206, 357, 479, 143, 451, 117, 461, 181, 80, 85, 347, 148, 269, 174, 443, 495, 475, 221, 151, 142, 304, 281, 423, 308, 188, 4, 385, 390, 453, 466, 70, 161, 122, 413, 234, 320, 113, 456, 241, 150, 165, 414, 236, 382, 386, 291, 490, 310, 156, 322, 283, 160, 194, 290, 198, 186, 191, 286, 268, 26, 480, 345, 28, 27, 439, 370, 255, 82, 417, 452, 22, 362, 169, 403, 488, 63, 331, 433, 201, 103, 242, 325, 146, 108, 140, 14, 248, 336, 102, 163, 36, 216, 244, 237, 89, 158, 494, 252, 396, 424, 12, 124, 111, 79, 259, 482, 249, 229, 373, 284, 251, 340, 454, 226, 328, 412, 243, 235, 447, 225, 25, 162, 141, 65, 272, 247, 136, 171, 180, 215, 155, 71, 197, 8, 192, 112, 187, 209, 415, 279, 372, 73, 467, 133, 478, 463, 7, 54, 18, 126, 205, 262, 483, 499, 130, 471, 77, 266, 416, 152, 486, 492, 457, 59, 263, 420, 369, 425, 138, 182, 203, 86, 360, 274, 354, 220, 233, 38, 298, 264, 265, 429, 484, 276, 418, 29, 115, 318, 33, 379, 39, 93, 91, 9, 232, 21, 400, 496, 267, 406, 376, 297, 438, 78, 154, 381, 450, 34, 184, 497, 159, 260, 105, 195]
    
    # Convert the sample array to a DataFrame
    df = spark.createDataFrame([(x,) for x in array], ["value"])
    
    # Arrange the array using parallel processing and quicksort within partitions
    arranged_array = arrange_array(df)

# Stop SparkSession
spark.stop()

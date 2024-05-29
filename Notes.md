<details>
     <summary>Overview of PSO</summary>
     
  ### Dataset: x, y, Class 
   
  -   x: [0,100] \
  -   y: [0,100] \
  -   Class: {A,B,C,D} - represent quadrants in graph
  
   | A    | B |
   | -------- | ------- |
   | C  | D    |

  -   PSO returns centroid - estimated middle of each quadrant.
  - 
  -   Training: split data into each class, run algorithm on each class
  
  ### Results
    
  -   Testing: For every data point in test data, find which centroid the data point is closest to. Corresponding class is the classificication. 
  -   Plotting results: While running, for each iteration, mark global best, calculate accuracy. Plot accuracy on graph (Accuracy vs. Iteration)
  -   When to stop?? when meets a convergence standard or max iterations
  -   Time? somehow calculate time

</details>
<details>
  <summary>Parrelization</summary>
  
  `myRDD = sc.parrelize()`

- not all of the spark context functions work
- rdd belongs to spark session
- rdd hard to print
  - use sql or .show() or other ways to print
- Same programs but one uses rdd other sc
  - PSO4Cluster.rdd
  - PSO4Cluster.sc

**mapPartition**
pass in rdd data
options:
1. split data among all nodes
 - data different between particles, but particles can all communicate between each other
2. split particles
- data same between all of particles, but can't communicate between each other
</details>

<details>
  <summary>My algorithm</summary>
  
  ### Island Algorithm
  Problem: Communicating between particles
  
  Solutions:
  
  - read and write with external file (hdfs? <- this might not be right, look more into this)
  - my idea: inside spark cluster add to accumulator (write only in master), use broadcast(read only in master) to send to inside spark cluster
</details>

<details>
  <summary>Things to keep in mind</summary>
  
  - Avoid examples that use spark context, as the spark context library isn’t configured well in the latest version of Pyspark.
  - Avoid using the Pyspark Shell.
  - Ensure a proper understanding of the form or type of data in your program, especially when reading the data exclusively, changing the datatype, converting to a different dataframe, or altering the schema.
  - broadcast variable is read only
    -broadcast should only be able to send from master to worker
  - accumulator : write only variable
    - `a = sc.accumulator(1)` look more on spark docs
  - within a spark task you cannot use any other spark functions
</details>

<details>
     <summary>Firefly algorithm</summary>
     
### Shared variables in spark
1. Accumulator
     - can be updated, value can only be accessed after .collect()
2. Broadcast variable
     - read only variable
- mapPartition(foo) divides function to all workers
     - foo is called a spark task
- until using .collect(), spark task won't end
###Firefly algorithm
- implemented in an island manner
- 4Cluster2DataSet.csv or genereate4Cluster2Ddataset.py
- Expected centroid values:
     - A: (70,30)
     - B: (70,70)
     - C: (30,30)
     - D: (30,70)
</details>

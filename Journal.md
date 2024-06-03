# Research Journal

<details>
  <summary>Week 1</summary>
  
  ### 5/21
  First day, intro to project and machine learning. 

  ### 5/22
  Setting up Apache Spark
  - Connecting to NDSU server
  - Running simple examples

  ### 5/23

  - overview of PSO with Aaron
  - begin working on parrelizing sorting algorithm
    - got a working program, but not sure if it was parrezling.
    - only used 2 partiitons on local machine, 1 on remote 
    

  ### 5/24
  - trying to fix sorting algorithm
    - previous code no longer working
    - simple square function also not working
</details>

<details>
  <summary>Week 2</summary>
  
  ### 5/28
  - Code now working locally - didn't make any changes.

  ### 5/29
  - Created  `firefly.py` algorithm without spark
    - created simple algorithm using chatgpt
    - developed `gridSearch.py` to find best parameters
    - on `4Cluster2D.csv` resulted in 100% accuracy (with rounding)

  ### 5/30
  - Altered `firefly.py` program to take in dataset with any number of classes or dimension. 
  - Worked on finding parameters that fit all programs, but with little sucesss.  `4Cluster2D.csv` is an ideal dataset, so the program converged perfectly. But other datasets did not consistently come within 10 units of error. 

  ### 5/31
  - Continued attempting to tweak paramaeters and `firefly.py` to improve accuracy. Added the following features:
    - Convergence condition: when improvement is consistently below a certain threshold, stop program and determine convergence.
    - Dynamic alpha: alpha level starts high and gradually decreases
  - Created an initial version of the FireFly algorithm with Spark. 
    - Using Spark requries handling the dataset in a slightly different way. 
    - New algorithm uses a Spark dataframe instead of pandas.  

</details>

<details>
  <summary>Week 3</summary>

  ### 6/31
    - Updated fireflyTest to include testing. It now takes a subset of data and classifies it using the model and returns an accuracy level.
      - For binary classification accuracy is 100%. 
      - For 4 classes the accuracy is between 92-95%. 
    - Worked on introductory presentation
</details>

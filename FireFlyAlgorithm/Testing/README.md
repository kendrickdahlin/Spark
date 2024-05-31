# Firefly Testing

This directory contains code to test the basic Firefly Algorithm. 

## Contents

1. **firefly.py**
   - This program uses the firefly algorithm to find the centroid of a dataset. The program inputs a csv file: In a row, the rightmost column contains the classification of a coordinate, and the other columns contain the coordinate points.
2. **fireflyTest.py**
    - This script can be easily modified to test different parameters in `firefly.py`
3. **generate2Cluster4D.py**
    - This script generates specific data that has two classes and four dimensions. The data is saved as a csv file in the working directory. The function `generate_data()` returns the ranges for the data. 
4. **generate4Cluster2D.py**
    - This script generates specific data that has four classes and four dimensions. The data is saved as a csv file in the working directory. The function `generate_data()` returns the ranges for the data. 
5. **generate2Cluster4D.py**
    - This script generates data with a given number of classes and dimension. The data is saved as a csv file in the working directory. The function `generate_data()` takes in the number of classes and dimension. returns the ranges for the data. 
6. **gridSearch.py**
    - This script searches for the best parameters for the FireFly algorithm given a dataset and it's true centers. 
7. **testingScores.txt**
    - Results from running `gridSearch.py`
8. **2Cluster4D.csv**
    - Data with two classes and four dimensions. True centers are:  {A: (25,25,25,25), B: (75,75,75,75)}
9. **4Cluster2D.csv**
    - Data with four classes and two dimensions. True centers are:  {A: (70,30), B: (70,70), C: (30,30), D: (30,70)}

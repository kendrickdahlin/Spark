import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import csv

def repeat_csv(output_file, repetitions):
    input_file = 'Behavior.csv'
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)    # Read the rest of the rows

    # Write to the output file
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header once
        
        for _ in range(repetitions):
            writer.writerows(rows)  # Write the rows repeatedly
            

for i in range(100
FilePath = 'Test.csv'
TestNum = 1
NumParticles = 50 
NumOfFold = 56
StartTime = time.time()

repeat_csv(FilePath,NumOfFold)
run(TestNum,FilePath,NumParticles,StartTime,TestList)
print(TestList)

spark.stop()
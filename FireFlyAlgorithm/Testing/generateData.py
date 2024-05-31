"""
Author: Kendrick Dahlin
"""
import random
import csv


"""
Given inputs creates a csv file of data for a swarm-intelligence program
Parameters: 
    - num_rows: number of data points   
    - num_classes: number of clusters
    - num_columns: dimenion of dataset
Return: Ranges of each class 
    - {'String': [(Int,Int), ... , (Int,Int)], ..., 'String': [(Int,Int) ... ,(Int,Int)]}
"""
def generate_data(num_rows, num_classes, num_columns):
    data = []
    categories = [chr(65 + i) for i in range(num_classes)]
    ranges = generate_ranges(num_classes, num_columns)

    for _ in range(num_rows):
        category = random.choice(categories)
        values = [random.uniform(min(r), max(r)) for r in ranges[category]]
        data.append(values + [category])

    save_to_csv(data, f"{num_classes}Cluster{num_columns}D.csv", num_columns)
    
    centers = {}
    for key, value in ranges.items(): 
        centers[key] = [(pair[0] + pair[1])/2 for pair in value]
    return ranges

def generate_ranges(num_classes, num_columns):
    min_value = 0
    max_value = 100

    step = (max_value - min_value) // num_classes
    ranges = {}
    for i in range(num_classes):
        category = chr(65 + i)
        ranges[category] = [
            (min_value + i * step, min_value + (i + 1) * step) for _ in range(num_columns)
        ]
    return ranges

def save_to_csv(data, filename, num_columns):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'x{i+1}' for i in range(num_columns)] + ['Class']
        writer.writerow(header)
        writer.writerows(data)


import random
import csv

def generate_data(num_rows):
    data = []
    for _ in range(num_rows):
        category = random.choice(['A', 'B', 'C', 'D'])
        if category == 'A':
            x = random.uniform(50, 90)
            y = random.uniform(10, 50)
        elif category == 'B':
            x = random.uniform(50, 90)
            y = random.uniform(50, 90)
        elif category == 'C':
            x = random.uniform(10, 50)
            y = random.uniform(10, 50)
        elif category == 'D':
            x = random.uniform(10, 50)
            y = random.uniform(50, 90)
        data.append([x, y, category])
    return data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'Class'])
        writer.writerows(data)

def main():
    num_rows = int(input("Enter the number of rows for the dataset: "))
    data = generate_data(num_rows)
    filename = "4Cluster2Ddataset.csv"
    save_to_csv(data, filename)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    main()

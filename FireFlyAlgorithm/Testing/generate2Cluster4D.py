import random
import csv

def generate_data(num_rows=100):
    num_classes = 2
    num_columns = 4
    data = []
    categories = [chr(65 + i) for i in range(num_classes)]
    ranges = {}

    #defining category ranges
    ranges['A'] = [(10, 40), (10,40), (10,40), (10,40)]
    ranges['B'] = [(60, 80), (60,80), (60,80), (60,80)]

    for _ in range(num_rows):
        category = random.choice(categories)
        values = [random.uniform(min(r), max(r)) for r in ranges[category]]
        data.append(values + [category])

    save_to_csv(data, f"{num_classes}Cluster{num_columns}D.csv", num_columns)
    centers = {}
    for key, value in ranges.items(): 
        centers[key] = [(pair[0] + pair[1])/2 for pair in value]
    return centers
def save_to_csv(data, filename, num_columns):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'x_{i+1}' for i in range(num_columns)] + ['Class']
        writer.writerow(header)
        writer.writerows(data)

def main():
    num_rows = int(input("Enter the number of rows for the dataset: "))
    num_classes = int(input("Enter the number of classes: "))
    num_columns = int(input("Enter the number of columns: "))
    data = generate_data(num_rows, num_classes, num_columns)
    filename = "Cluster2Ddataset.csv"
    save_to_csv(data, filename, num_columns)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    main()

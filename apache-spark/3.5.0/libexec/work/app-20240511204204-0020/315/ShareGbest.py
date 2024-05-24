import csv

def ReadGbestValue():
    try:
        with open('/opt/homebrew/Cellar/apache-spark/3.5.0/libexec/Gbest.csv', mode='r') as file:
            reader = csv.reader(file)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return list(map(float, last_row))
            else:
                return None
    except FileNotFoundError:
        return None

def WriteGbestValue(x, y):
    with open('/opt/homebrew/Cellar/apache-spark/3.5.0/libexec/Gbest.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([x, y])

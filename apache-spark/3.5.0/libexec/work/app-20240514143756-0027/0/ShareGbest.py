import csv

import csv

def ReadGbestValue():
    try:
        with open('Gbest.csv', 'r') as file:
            reader = csv.reader(file)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return float(last_row[0]), float(last_row[1])
            else:
                return None
    except FileNotFoundError:
        return None

def WriteGbestValue(x, y, Cid, Ittr):
    with open('/opt/homebrew/Cellar/apache-spark/3.5.0/libexec/Gbest.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([x, y, Cid, Ittr])


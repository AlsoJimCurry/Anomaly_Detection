import random
import csv

with open("data.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["priority", "latency", "jitter", "framelength"])
    
    for i in range(1,500):
        writer.writerow([1, random.uniform(0.3, 0.8), random.uniform(0, 0.05), 64])
    
    for i in range(1,500):
        writer.writerow([5, random.uniform(0.3, 1.2), random.uniform(0, 0.05), 64])
 
    for i in range(1,700):
        writer.writerow([10, random.uniform(0.3, 5), random.uniform(0, 1), 64])
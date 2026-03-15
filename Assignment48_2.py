import pandas as pd
import numpy as np

data=[6, 7, 8, 9, 10, 11, 12]

print("Varience is :",np.var(data))
print("Standard Deviation is :",np.std(data))

# manually

n=len(data)

mean=sum(data)/n

variance = sum((x-mean)**2 for x in data)/n

std_dev = variance**0.5
print("manual Mean:",mean)
print("manual Variance:",variance)
print("manual Standard Deviation:",std_dev)
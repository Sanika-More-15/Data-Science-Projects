import numpy as np
from sklearn.metrics import confusion_matrix

actual = [1, 1, 1, 1, 0, 0, 0, 0]
predicted = [1, 1, 0, 1, 0, 1, 0, 0]

cm = confusion_matrix(actual, predicted)

tn, fp, fn, tp = cm.ravel() #Converts dimensional arrays into a 1D structure.

print("Confision Metrix :")
print(cm)

print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
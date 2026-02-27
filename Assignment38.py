import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Border="-"*40
################################################################################################
# Step 1 : Write a Python program to load the final student_performance_ml.csv using pandas.
################################################################################################
print(Border)
print("Step 1 : A Python program to load the final student_performance_ml.csv using pandas.")
print(Border)

Dataset="student_performance_ml.csv"
df=pd.read_csv(Dataset)
print("Dataset get's loaded successfully\n")
print("First 5 records from dataset :")
print(df.head(),"\n")
print("Last 5 records from dataset :")
print(df.tail(),"\n")
print("Total number of rows and columns :",df.shape)
print("\nList of column names :",list(df.columns))
print("\nDatatype of each column :")
print(df.dtypes)

################################################################################################
# Step 2 : Display total number of students in dataset, Count how many stunedts passed and failed.
################################################################################################
print(Border)
print("Step 2 : Total number of students in dataset, Count how many stunedts passed and failed.")
print(Border)

Total_students=df.shape[0]
print("Total number of students :",Total_students)

Result_count=df['FinalResult'].value_counts()

print("\nstudent Result Count :")
print(Result_count)

print("\nNumber of Passed Students :",Result_count[1])
print("Number of failed Students :",Result_count[0])

################################################################################################
# Step 3 : Using pandas function, calculate and display: Average StudyHours, Average Attendence, Maximum PreviousScore, Minimum SleepHours
################################################################################################
print(Border)
print("Step 3 : Average StudyHours, Average Attendence, Maximum PreviousScore, Minimum SleepHours")
print(Border)

#print(df.describe())
Avg_Studyhours=df['StudyHours'].mean()
print("Average Study Hours of Students :",Avg_Studyhours)

Avg_Attendence=df['Attendance'].mean()
print("Average Attendence of Students :",Avg_Attendence)

Max_PreviousScore=df['PreviousScore'].max()
print("Maximum Previsous Score of students :",Max_PreviousScore)

Min_SleepHours=df['SleepHours'].min()
print("Minimum Sleep hours of students :",Min_SleepHours)

################################################################################################
# Step 4 : Use value_count() to analyse the distribution of FinalResult.
# Calculate the percentage of Pass and fail students. Is the dataset balanced ? 
################################################################################################
print(Border)
print("Step 4 : Distribution of  FinalResult")
print(Border)

Result=df['FinalResult'].value_counts()

print("Student Distribution :")
print(Result)

#Percentage Calculation
Percent=df['FinalResult'].value_counts(normalize=True)*100
print("\nPercentage :",Percent)

#Check balance
if Percent.max() - Percent.min() <= 10:
    print("Dataset is Balanced")
else:
    print("Dataset is Imbalanced")

################################################################################################
# Step 5 : Analyse whether : Highter StydyHours increase the chance of passing.
# Higher Attendence improves FinalResult.
################################################################################################
print(Border)
print("Step 5 : Effect of StudyHours and Attendance on FinalResult")
print(Border)

# Average StudyHours based on result
print("\nAverage StudyHours:")
print(df.groupby("FinalResult")["StudyHours"].mean())

# Average Attendance based on result
print("\nAverage Attendance:")
print(df.groupby("FinalResult")["Attendance"].mean())

################################################################################################
# Step 6 : Histogram of StudyHours
################################################################################################
print(Border)
print("Step 6 : Histogram of StudyHours")
print(Border)

sns.histplot(df["StudyHours"], bins=10, kde=True)
plt.title("StudyHours Distribution")
plt.show()

################################################################################################
# Step 7 : Scatterplot of StudyHours VS PreviousScore
################################################################################################
print(Border)
print(" Step 7 : Scatterplot of StudyHours VS PreviousScore")
print(Border)

sns.scatterplot(x='StudyHours',y='PreviousScore',hue='FinalResult',data=df)
plt.show()

################################################################################################
# Step 8 : Boxplot for Attendence
################################################################################################
print(Border)
print("Step 8 : Boxplot for Attendence")
print(Border)

sns.boxplot(df["Attendance"])
plt.show()

################################################################################################
# Step 9 : Relationship between AssignmentCompleted and Final Result
################################################################################################
print(Border)
print("Step 9 : Relationship between AssignmentCompleted and Final Result")
print(Border)

sns.boxplot(x='FinalResult',y='AssignmentsCompleted',data=df)
plt.title("AssignmentCompleted VS FinalResult")
plt.show()

################################################################################################
# Step 10 : SleepHour Vs FinalResult
################################################################################################
print(Border)
print("Step 10 : SleepHour Vs FinalResult")
print(Border)

sns.barplot(x='FinalResult',y='SleepHours',data=df)
plt.title("SleepHour VS FinalResult")
plt.show()


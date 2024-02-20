import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv('employee_data.csv')

#bucketing/isnumeric conversion 
df["MonthlyIncome"].replace(to_replace="low", value=1, inplace=True)
df["MonthlyIncome"].replace(to_replace="medium", value=2, inplace=True)
df["MonthlyIncome"].replace(to_replace="high", value=3, inplace=True)

df["Left"].replace(to_replace="No", value=0, inplace=True)
df["Left"].replace(to_replace="Yes", value=1, inplace=True)

df["Gender"].replace(to_replace="Female", value=0, inplace=True)
df["Gender"].replace(to_replace="Male", value=1, inplace=True)

df["BusinessTravel"].replace(to_replace="Non-Travel", value=0, inplace=True)
df["BusinessTravel"].replace(to_replace="Travel_Rarely", value=1, inplace=True)
df["BusinessTravel"].replace(to_replace="Travel_Frequently", value=2, inplace=True)

#create correlation values
corr = df.corr()

#create heatmap with values
sns.heatmap(corr, annot=True, cmap='YlGnBu')


pearsonr(df)



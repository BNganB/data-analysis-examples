import pandas as pd
from sklearn import linear_model
from scipy.stats import pearsonr

df = pd.read_csv('employee_data2.csv')

#split data into training groups
train=df.sample(frac=0.8, random_state=200)
test=df.drop(train.index)

#convert string values into numeric for correlation comparison
train["MonthlyIncome"].replace(to_replace="low", value=1, inplace=True)
train["MonthlyIncome"].replace(to_replace="medium", value=2, inplace=True)
train["MonthlyIncome"].replace(to_replace="high", value=3, inplace=True)

train["Left"].replace(to_replace="No", value=0, inplace=True)
train["Left"].replace(to_replace="Yes", value=1, inplace=True)

train["Gender"].replace(to_replace="Female", value=0, inplace=True)
train["Gender"].replace(to_replace="Male", value=1, inplace=True)

train["BusinessTravel"].replace(to_replace="Non-Travel", value=0, inplace=True)
train["BusinessTravel"].replace(to_replace="Travel_Rarely", value=1, inplace=True)
train["BusinessTravel"].replace(to_replace="Travel_Frequently", value=2, inplace=True)

#change null values to 0 to allow sklearn to perform linear regression
train.fillna(0, inplace=True)

#multiple regression function creation
X = train[['Age', #weak negative correlation, indicates younger employees are more likely to leave the company
        'Gender',
        'MonthlyIncome', #weak negative correlation, potentially spurious
        'NumCompaniesWorked',
        'workingfromhome',
        'BusinessTravel', #less significant, but shows >0.1 positive correlation
        'DistanceFromHome',
        'JobSatisfaction', #slight negative correlation, unexpected but insignificant
        'complaintfiled',
        'complaintyears', #highly weighted indicator for potential second model
        'PercentSalaryHike',
        'PerformanceRating',
        'TotalWorkingYears',
        'YearsAtCompany']]
y = train['Left']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedleaver = regr.predict([[X, y]])

#Closer value is to 1, more likely employee is to leave company
print(predictedleaver)

#Obtain pearson r-value for two columns independently
pearsonr(column1, column2)
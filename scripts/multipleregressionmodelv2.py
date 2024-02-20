import pandas as pd
from sklearn import linear_model

df = pd.read_csv('employee_data2.csv')

#convert string values into numeric for correlation comparison
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

#change null values to 0 to allow sklearn to perform linear regression
df.fillna(0, inplace=True)

#split data into training groups
train=df.sample(frac=0.8, random_state=200)
test=df.drop(train.index)

#multiple regression function creation
X = train[['Age', #weak negative correlation, indicates younger employees are more likely to leave the company
        'MonthlyIncome', #weak negative correlation, potentially spurious
        'BusinessTravel', #less significant, but shows >0.1 positive correlation
        'complaintyears', #highly weighted indicator for potential second model
        'TotalWorkingYears',
        'YearsAtCompany']]
y = train['Left']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#Obtain coef values for all independent variables
print(regr.coef_)

#strip unneeded variables from test case
test_refined = test.drop(("Gender", "Department", "NumCompaniesWorked", "Over18", "workingfromhome", "DistanceFromHome", "StandardHours", "JobSatisfaction", "complaintfiled", "complaintresolved", "PercentSalaryHike", "PerformanceRating", "YearsSinceLastPromotion"), axis="columns")

#model trained on training data as not to overfit
#Value range 0-1, closer to 1 = more likely to leave
predictedleaver = regr.predict([[test_refined.sample(n=1)]])
print(predictedleaver)
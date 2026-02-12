# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
##Date-12/02/2026
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kamaleshwaran S
RegisterNumber: 212225040165
*/
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/content/Placement_Data.csv')
df1 = df.copy()

# Drop unnecessary columns
df1 = df1.drop(['sl_no', 'salary'], axis=1)

# Check for missing values and duplicates
df1.isnull().sum()
df1.duplicated().sum()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])

# Split features and target
x = df1.iloc[:, :-1]
y = df1['status']

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)

# Display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
cn = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['true', 'false'])
cn.plot()
```

## Output:
<img width="940" height="1004" alt="image" src="https://github.com/user-attachments/assets/1fd4a608-9b03-4ba7-b7db-d3990ddc57ce" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

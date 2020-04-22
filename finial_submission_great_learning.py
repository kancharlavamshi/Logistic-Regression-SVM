
###### Lgogistic Regression  ###########3
### mounting google drive ,because iam executing this code in colab
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/project-GL/diabetes.csv')
data.head()
### describing the data
data.describe().T

data1 = data.copy(deep=True)
data1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
###  making the null value to mean and median
data1['Glucose'].fillna(data1['Glucose'].mean(), inplace = True)
data1['BloodPressure'].fillna(data1['BloodPressure'].mean(), inplace = True)
data1['SkinThickness'].fillna(data1['SkinThickness'].median(), inplace = True)
data1['Insulin'].fillna(data1['Insulin'].median(), inplace = True)
data1['BMI'].fillna(data1['BMI'].mean(), inplace = True)
### displaying the data  
p = data1.hist(figsize = (20,20), rwidth=0.9)

x = data1[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age']].values
y = data1[['Outcome']].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))



#######  support vector machine ##############

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
data_svm = pd.read_csv('/content/drive/My Drive/Colab Notebooks/project-GL/diabetes.csv')
data_svm.head()

X = data_svm.iloc[:,0:9].values
y = data_svm.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
cfr = SVC(kernel = 'rbf', random_state = 0)
cfr.fit(X_train, y_train)
y_pred = cfr.predict(X_test)
accuracies = cross_val_score(estimator = cfr, X = X_train, y = y_train, cv = 10)
accuracies.mean()

###############  Naive Bayes Algorithm  #####################

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
dat = pd.read_csv('/content/drive/My Drive/Colab Notebooks/project-GL/diabetes.csv')
data_nb = dat
data_nb.head()

X = data_nb.iloc[:,0:9].values
y = data_nb.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
model1=GaussianNB()
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)

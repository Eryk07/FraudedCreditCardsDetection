import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

ccdata = pd.read_csv('creditcard.csv')

#print(ccdata.head())   #first five rows of data
#print(ccdata.columns)  #atributes of dataframe
#print(ccdata.tail())   #last five rows of data
#print(ccdata.describe())   #summary of dataframe

fig1, ax1 = plt.subplots()
ax1.set_title('Features boxplot')
ax1.boxplot(ccdata, notch=True)
plt.show()

#scaling column 'Amount'
scaler = MinMaxScaler()
sccdata = ccdata
sccdata[['Amount']] = scaler.fit_transform(ccdata[['Amount']])

#split into train and test datasets
X = sccdata.iloc[:, 0:30].values
y = sccdata.iloc[:, 30].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8) #80:20 proportions

### A) Logistic regression ###
logistic_model = LogisticRegression(solver='lbfgs')
logistic_model.fit(X_train, y_train)
y_predict_log = logistic_model.predict(X_test)
y_predict_log_rounded = [np.round(value) for value in y_predict_log]
accuracy_log = accuracy_score(y_test, y_predict_log_rounded)

### B) Decision Tree ###
tree_model = DecisionTreeClassifier(random_state=1)
tree_model.fit(X_train, y_train)
y_predict_tree = tree_model.predict(X_test)
y_predict_tree_rounded = [np.round(value) for value in y_predict_tree]
accuracy_tree = accuracy_score(y_test, y_predict_tree_rounded)

### C) Artificial Neural Network ###
ANN_model = MLPClassifier(random_state=1)
ANN_model.fit(X_train, y_train)
y_predict_ANN = ANN_model.predict(X_test)
y_predict_ANN_rounded = [np.round(value) for value in y_predict_ANN]
accuracy_ANN = accuracy_score(y_test, y_predict_ANN_rounded)

### D) Gradient Boosting ###
GBM_model = GradientBoostingClassifier(random_state=1)
GBM_model.fit(X_train, y_train)
y_predict_GBM = GBM_model.predict(X_test)
y_predict_GBM_rounded = [np.round(value) for value in y_predict_GBM]
accuracy_GBM = accuracy_score(y_test, y_predict_GBM_rounded)

### Summary ###
print('Logistic Regression: ', accuracy_log)
print('Decision Tree', accuracy_tree)
print('Artificial Neural Network', accuracy_ANN)
print('Gradient Boosting', accuracy_GBM)

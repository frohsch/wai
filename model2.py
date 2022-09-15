import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
data = pd.read_csv('C:/Users/WIN10/Desktop/merge3.csv', encoding='cp949')

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = data.loc[:,'avgTemp':'humidity']
x = x.astype(int)

y = data.loc[:,'power']
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#Initialize regressor
regressor = LinearRegression()

#Begin fitting the training sets
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)
predicted = regressor.predict(X_test)
#plt.scatter(y_test, predicted)

#scatterplot of training set
#predicted = regressor.predict(X_train)
#plt.scatter(y_train, predicted)
#Save model
import pickle
pickle.dump(regressor, open('model3.pkl', 'wb'))
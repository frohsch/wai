

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score # 정확도 함수
import pandas as pd
data = pd.read_csv('C:/Users/WIN10/Desktop/merge3.csv', encoding='cp949')

from sklearn.model_selection import KFold, cross_val_score, train_test_split

x = data.loc[:,'avgTemp':'humidity']
x = x.astype(int)

y = data.loc[:,'power']
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)



#Initialize regressor

regressor1 = RandomForestRegressor(n_estimators=20, random_state=0)

#Begin fitting the training sets
regressor1.fit(X_train, y_train)

predicted = regressor1.predict(X_test)
#plt.scatter(y_test, predicted)

#scatterplot of training set
#plt.scatter(y_train, predicted)
#Save model
import pickle
pickle.dump(regressor1, open('model4.pkl', 'wb'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# Get the data
train_X = pd.read_csv("./train.csv", index_col="Id")
test_X = pd.read_csv("./test.csv", index_col="Id")

# Select numerical features and examples with missing values
train_X = train_X.select_dtypes(include=np.number).dropna()
test_X = test_X.select_dtypes(include=np.number).fillna(0)

# Move SalePrice to target data variable
train_X, train_y = train_X.iloc[:, :-1], train_X.iloc[:, -1]

# Train a model
clf = RandomForestRegressor(random_state=0)
clf.fit(train_X, train_y)

# Get predictions from the model
predictions = clf.predict(test_X)

# Get the solution data
solution = pd.read_csv('./solution.csv')
y_true = solution["SalePrice"]

# Calculate the score right here
RMSLE = np.sqrt(mean_squared_log_error(y_true, predictions))
print("The score is %.5f" % RMSLE)

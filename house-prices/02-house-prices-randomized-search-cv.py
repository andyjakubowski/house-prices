import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

# Get the data
X = pd.read_csv("./train.csv", index_col="Id")

# Select numerical features and examples with missing values
X = X.select_dtypes(include=np.number).dropna()

# Move SalePrice to target data variable
X, y = X.iloc[:, :-1], X.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

param_distributions = {
    "n_estimators": randint(50, 150),
    "max_depth": randint(5, 20)
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    n_iter=20,
    param_distributions=param_distributions,
    random_state=0,
    scoring='neg_mean_squared_log_error'
)

# Train a model
search.fit(X_train, y_train)

print(f"search.best_params_: {search.best_params_}")
score = search.score(X_test, y_test)
print(f"search.score(): {score}")

# Get competition test data
X_competition = pd.read_csv("./test.csv", index_col="Id")

# Process test data
X_competition = X_competition.select_dtypes(include=np.number).fillna(0)

# Get predictions from the model
predictions = search.predict(X_competition)
submission = pd.DataFrame(
    predictions, index=X_competition.index, columns=["SalePrice"])
print(f"submission: {submission}")

# Save submission to CSV file
submission.to_csv("submission_02.csv")

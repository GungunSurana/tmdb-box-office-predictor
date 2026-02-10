#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[ ]:


# Load data
DATA_PATH = "/kaggle/input/dataset1/"

train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

# Detect ID column
id_col = "movie_id" if "movie_id" in test.columns else "id"


# In[ ]:


# Feature engineering
for df in [train, test]:
    df["release_date"] = pd.to_datetime(
        df["release_date"], errors="coerce", dayfirst=True
    )
    df["release_year"] = df["release_date"].dt.year

def genre_count(x):
    return len(str(x).split(",")) if pd.notna(x) else 0

train["genre_count"] = train["genres"].apply(genre_count)
test["genre_count"] = test["genres"].apply(genre_count)

# Features & target
features = [
    "budget",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "release_year",
    "genre_count"
]

X = train[features]
y = np.log1p(train["revenue"])
X_test = test[features]


# In[ ]:


# Handle missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Train 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

rmse_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmse_scores[name] = rmse
    print(f"{name} RMSE:", rmse)
# Pick best model
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

print("\nBest model selected:", best_model_name)

# Train best model on full data
best_model.fit(X, y)
final_preds = np.expm1(best_model.predict(X_test))

# Create submission file
submission = pd.DataFrame({
    "movie_id": test[id_col],
    "revenue": final_preds
})


# In[ ]:


submission.to_csv("/kaggle/working"+ "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Plot histogram of actual revenue
plt.figure()
plt.hist(train["revenue"], bins=50)
plt.xlabel("Revenue")
plt.ylabel("Number of Movies")
plt.title("Distribution of Movie Revenue")
plt.show()


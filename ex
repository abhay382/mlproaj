# ============================================================
# MACHINE LEARNING REGRESSION PROJECT (Mercedes Car Price)
# Complete Step-by-Step with Comments
# ============================================================

# ============================================================
# STEP 1 : Import Required Libraries
# ============================================================

import pandas as pd                 # Used for data manipulation
import numpy as np                  # Used for mathematical operations
import matplotlib.pyplot as plt     # Used for plotting graphs

# ============================================================
# STEP 2 : Load Dataset
# ============================================================

df = pd.read_csv("merc.csv")        # Read CSV file into DataFrame

# ============================================================
# STEP 3 : Basic Data Inspection
# ============================================================

df.head()                           # Display first 5 rows

df.tail()                           # Display last 5 rows

df.shape                            # Returns (Rows, Columns)

df.columns                          # Shows all column names

df.info()                           # Data types, null values, memory usage

df.describe()                       # Statistical summary of numerical columns

# ============================================================
# STEP 4 : Data Cleaning
# ============================================================

# Check missing values
df.isnull().sum()

# Fill missing numerical values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check duplicate rows
df.duplicated().sum()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# ============================================================
# STEP 5 : Exploratory Data Analysis (EDA)
# ============================================================

# Histogram of Car Price
df["price"].hist()

plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Correlation between numerical columns
df.corr(numeric_only=True)

# ============================================================
# STEP 6 : Outlier Detection
# ============================================================

# Calculate First Quartile (25%)
Q1 = df["price"].quantile(0.25)

# Calculate Third Quartile (75%)
Q3 = df["price"].quantile(0.75)

# Calculate Inter Quartile Range
IQR = Q3 - Q1

# Lower Boundary
Lower = Q1 - 1.5 * IQR

# Upper Boundary
Upper = Q3 + 1.5 * IQR

# Display Outliers
outliers = df[(df["price"] < Lower) | (df["price"] > Upper)]

# Optional: Remove Outliers
# df = df[(df["price"] >= Lower) & (df["price"] <= Upper)]

# ============================================================
# STEP 7 : Feature Engineering
# ============================================================

# Create new feature: Car Age
df["car_age"] = 2026 - df["year"]

# Create new feature: Mileage Per Year
df["mileage_per_year"] = df["mileage"] / df["car_age"]

# ============================================================
# STEP 8 : Define Features and Target
# ============================================================

# Features (Independent Variables)
X = df.drop("price", axis=1)

# Target Variable (Dependent Variable)
y = df["price"]

# ============================================================
# STEP 9 : Split Dataset
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,          # 20% Testing Data

    random_state=42          # Same split every time

)

# ============================================================
# STEP 10 : Data Preprocessing
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Categorical Columns
categorical_features = [

    "model",

    "transmission",

    "fuelType"

]

# One Hot Encoding converts text into numbers
preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        )

    ],

    remainder="passthrough"

)

# Learn categories and transform training data
X_train = preprocessor.fit_transform(X_train)

# Transform testing data
X_test = preprocessor.transform(X_test)

# ============================================================
# STEP 11 : Build Machine Learning Model
# ============================================================

from sklearn.ensemble import RandomForestRegressor

# Create Model
model = RandomForestRegressor(

    n_estimators=100,         # Number of Decision Trees

    random_state=42

)

# Train Model
model.fit(X_train, y_train)

# ============================================================
# STEP 12 : Prediction
# ============================================================

# Predict prices
y_pred = model.predict(X_test)

print(y_pred)

# ============================================================
# STEP 13 : Evaluate Model
# ============================================================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, y_pred)

# ============================================================
# STEP 14 : Print Results
# ============================================================

print("Mean Absolute Error :", mae)

print("Mean Squared Error :", mse)

print("Root Mean Squared Error :", rmse)

print("R2 Score :", r2)

# ============================================================
# END OF PROJECT
# ============================================================

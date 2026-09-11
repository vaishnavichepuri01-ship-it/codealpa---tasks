# ==========================================
# TASK 4: SALES PREDICTION USING PYTHON
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Advertising.csv")

print("First 5 rows:")
print(df.head())


# ==========================================
# 2. DATA EXPLORATION
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 3. DATA CLEANING
# ==========================================

# Remove unnecessary index column
df = df.drop("Unnamed: 0", axis=1)

print("\nColumns after cleaning:")
print(df.columns)


# ==========================================
# 4. FEATURE SELECTION
# ==========================================

X = df[
    [
        "TV",
        "Radio",
        "Newspaper"
    ]
]

Y = df["Sales"]


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# ==========================================
# 6. CREATE REGRESSION MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, Y_train)

print("\nModel training completed!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\nActual Sales:")
print(Y_test.values)

print("\nPredicted Sales:")
print(predictions)


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    Y_test,
    predictions
)

rmse = mean_squared_error(
    Y_test,
    predictions
) ** 0.5

r2 = r2_score(
    Y_test,
    predictions
)

print("\nModel Evaluation:")

print("MAE :", mae)

print("RMSE:", rmse)

print("R² Score:", r2)


# ==========================================
# 10. VISUALIZATION
# ==========================================

plt.scatter(
    Y_test,
    predictions
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.grid(True)

plt.show()


# ==========================================
# 11. ADVERTISING IMPACT
# ==========================================

print("\nAdvertising Impact:")

print("TV coefficient:", model.coef_[0])
print("Radio coefficient:", model.coef_[1])
print("Newspaper coefficient:", model.coef_[2])


# ==========================================
# 12. BUSINESS INSIGHTS
# ==========================================

print("\nBusiness Insights:")

print("- Advertising spending can be used to predict sales.")

print("- TV, Radio and Newspaper spending have different effects on sales.")

print("- The regression coefficients show the direction and strength of each advertising channel.")

print("- Businesses can use these results to make better advertising budget decisions.")
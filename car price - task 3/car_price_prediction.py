# ==========================================
# TASK 3: CAR PRICE PREDICTION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("car data.csv")

print("First 5 rows:")
print(df.head())


# ==========================================
# 2. EXPLORE THE DATA
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

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

# Create car age
current_year = 2026

df["Car_Age"] = current_year - df["Year"]

print("\nData after feature engineering:")
print(df.head())


# ==========================================
# 5. SELECT FEATURES AND TARGET
# ==========================================

X = df[
    [
        "Year",
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age"
    ]
]

Y = df["Selling_Price"]


# ==========================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
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
# 7. PREPROCESSING
# ==========================================

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 8. CREATE REGRESSION MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

model.fit(X_train, Y_train)

print("\nModel training completed!")


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\nActual Prices:")
print(Y_test.values)

print("\nPredicted Prices:")
print(predictions)


# ==========================================
# 11. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(Y_test, predictions)

rmse = mean_squared_error(
    Y_test,
    predictions
) ** 0.5

r2 = r2_score(Y_test, predictions)

print("\nModel Evaluation:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


# ==========================================
# 12. VISUALIZATION
# ==========================================

plt.scatter(
    Y_test,
    predictions
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.grid(True)
plt.show()


# ==========================================
# 13. PREDICT PRICE FOR A NEW CAR
# ==========================================

new_car = pd.DataFrame({
    "Year": [2020],
    "Present_Price": [8.5],
    "Driven_kms": [30000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0],
    "Car_Age": [2026 - 2020]
})

predicted_price = model.predict(new_car)

print("\nPredicted price for new car:")
print(predicted_price[0])
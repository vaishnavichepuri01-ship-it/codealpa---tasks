import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("archive/Unemployment_Rate_upto_11_2020.csv")

# 2. Data cleaning
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# 3. Data exploration
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nAverage Unemployment Rate:")
print(df["Estimated Unemployment Rate (%)"].mean())

# 4. Monthly unemployment trend
monthly_unemployment = df.groupby("Date")[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nMonthly Unemployment Rate:")
print(monthly_unemployment)

# 5. Visualize unemployment trend
plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Monthly Unemployment Rate in India - 2020")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 6. COVID-19 impact analysis
covid_period = df[
    (df["Date"] >= "2020-04-01") &
    (df["Date"] <= "2020-05-31")
]

covid_unemployment = covid_period[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nAverage Unemployment Rate during April-May 2020:")
print(covid_unemployment)

# 7. Identify key patterns
highest_month = monthly_unemployment.idxmax()
highest_rate = monthly_unemployment.max()

lowest_month = monthly_unemployment.idxmin()
lowest_rate = monthly_unemployment.min()

print("\nKey Patterns:")
print("Highest unemployment month:", highest_month)
print("Highest unemployment rate:", highest_rate)

print("Lowest unemployment month:", lowest_month)
print("Lowest unemployment rate:", lowest_rate)

# 8. Insights
print("\nInsights:")
print("- Unemployment increased sharply during April-May 2020.")
print("- This period coincided with the major COVID-19 lockdown period.")
print("- Unemployment declined considerably after May 2020.")
print("- The data shows a strong short-term impact of COVID-19 on unemployment.")
print("- Such patterns can help policymakers plan employment support during economic disruptions.")
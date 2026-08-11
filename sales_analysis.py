# Mini Porject: Sales Analysis and Forecasting:
# Smart Sales Performance Analytics & Forecasting Dashboard

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("data/sales_data.csv")
df.head()
df.info()
df.describe()

# Data Cleaning:
df.isnull().sum()
df.drop_duplicates(inplace=True)
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

# Feature Engineering:

df["Month"]=df["Sale_Date"].dt.month_name()
df["Quarter"]=df["Sale_Date"].dt.quarter
df["Year"]=df["Sale_Date"].dt.year
df["Revenue"]=df["Sales_Amount"]
print(df.head())

# Exploratory Data Analysis:

df["Revenue"].sum()
df["Revenue"].mean()
df["Revenue"].max()
df["Revenue"].min()

# Revenue by Region:

region=df.groupby("Region")["Revenue"].sum()

region.plot(kind="bar")

plt.title("Revenue by Region")

plt.show()

# Monthly sales trends:

monthly=df.groupby("Month")["Revenue"].sum()

monthly.plot(marker="o")

plt.title("Monthly Sales Trend")

plt.show()

# Product vs region Heatmap:

pivot=df.pivot_table(values="Revenue",
                     index="Product_Category",
                     columns="Region",
                     aggfunc="sum")

sns.heatmap(pivot,
            annot=True,
            cmap="YlGnBu")
plt.show()

# Pie Chart - Category Contribution;

category = df.groupby("Product_Category")["Revenue"].sum()

plt.pie(category,
        labels=category.index,
        autopct="%1.1f%%")

plt.title("Category Contribution")

plt.show()

# Top Products:

top=df.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False).head(10)

top.plot(kind="bar")
plt.show()

# Region Wise Sales:

region_sales=df.groupby("Region")["Revenue"].sum()
region_sales.plot(kind="bar")
plt.title("Region Wise Sales")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.show()

# Machine Learning Forecast:

x = df[["Quantity_Sold", "Discount"]]
y=df["Revenue"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

pred=model.predict(x_test)

# Forecast Graph:
# Create Month columns
df["Month_Num"] = pd.to_datetime(df["Sale_Date"]).dt.month
df["Month_Name"] = pd.to_datetime(df["Sale_Date"]).dt.strftime("%b")

# Monthly Actual Revenue
monthly_actual = (
    df.groupby(["Month_Num", "Month_Name"])["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Month_Num")
)

# Monthly Predicted Revenue
df["Predicted_Revenue"] = model.predict(x)

monthly_predicted = (
    df.groupby(["Month_Num", "Month_Name"])["Predicted_Revenue"]
    .mean()
    .reset_index()
    .sort_values("Month_Num")
)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_actual["Month_Name"],
    monthly_actual["Revenue"],
    marker="o",
    label="Actual Revenue"
)

plt.plot(
    monthly_predicted["Month_Name"],
    monthly_predicted["Predicted_Revenue"],
    marker="o",
    label="Predicted Revenue"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
import mysql.connector

# Read CSV file
df = pd.read_csv("data/sales_data.csv")

print("CSV data loaded successfully!")
print(df.head())
print("Total rows:", len(df))

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="system",
    database="smart_sales_analytics"
)

cursor = conn.cursor()

print("MySQL connected successfully!")

# Insert data into sales table
sql = """
INSERT INTO sales
(Order_id, date, Product_category, Product, Region, Quantity, Revenue)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

data = []

for _, row in df.iterrows():
    data.append((
        int(row["Order_id"]),
        pd.to_datetime(row["date"]).date(),
        row["Product_category"],
        row["Product"],
        row["Region"],
        int(row["Quantity"]),
        float(row["Revenue"])
    ))

cursor.executemany(sql, data)

conn.commit()

print("Data inserted successfully!")
print("Rows inserted:", cursor.rowcount)

cursor.close()
conn.close()

print("MySQL connection closed.")
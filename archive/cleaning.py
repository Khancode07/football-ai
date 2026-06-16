import pandas as pd
import sqlite3

# Database connect
conn = sqlite3.connect("football.db")

# Data load
df = pd.read_sql("SELECT * FROM matches", conn)

print("Pehle:", df.shape)

# Missing values check
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates check
print("\nDuplicates:", df.duplicated().sum())

# Cleaning
df = df.drop_duplicates()

# Date convert
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Invalid dates remove
df = df.dropna(subset=["date"])

# Invalid scores remove
df = df[df["home_score"] >= 0]
df = df[df["away_score"] >= 0]

# 1950 ke baad ke matches
df = df[df["date"].dt.year >= 1950]

print("\nBaad Mein:", df.shape)

# Clean table save
df.to_sql("matches_clean", conn, if_exists="replace", index=False)

print("\n✅ Cleaning Complete!")
print("✅ Data saved as matches_clean")

conn.close()
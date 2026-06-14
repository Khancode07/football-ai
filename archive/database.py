import pandas as pd
import sqlite3
import os

# CSV file ka naam
csv_file = "results (2).csv"

# Check file exists
if not os.path.exists(csv_file):
    print(f"❌ File nahi mili: {csv_file}")
    print("Folder ki files:")
    print(os.listdir())
    exit()

# CSV load
df = pd.read_csv(csv_file)
print(f"✅ Loaded: {df.shape[0]} rows")

# Data cleaning
df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['date'])

print(f"✅ Cleaned: {df.shape[0]} rows")

# SQLite database create
conn = sqlite3.connect("football.db")

# Table save
df.to_sql("matches", conn, if_exists="replace", index=False)

# Verify rows
count = pd.read_sql(
    "SELECT COUNT(*) AS total_rows FROM matches",
    conn
)

print(f"✅ Database Saved!")
print(f"✅ Rows in database: {count['total_rows'][0]}")

# Check tables
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)

print("\n📋 Tables:")
print(tables)

conn.close()

print("\n🎉 DONE!")
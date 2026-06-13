import sqlite3
import pandas as pd

conn = sqlite3.connect("football.db")

df = pd.read_sql(
    "SELECT * FROM matches_clean LIMIT 5",
    conn
)

print(df)

conn.close()
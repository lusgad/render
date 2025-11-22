# create_db.py
import pandas as pd
import sqlite3

def create_database():
    print("Loading data...")
    df = pd.read_csv('used.zip')
    print(f"Loaded {len(df)} rows")
    
    # Basic preprocessing
    df["CRASH_DATETIME"] = pd.to_datetime(df["CRASH_DATETIME"], errors="coerce")
    df["YEAR"] = df["CRASH_DATETIME"].dt.year
    
    # Create totals
    injury_cols = [col for col in df.columns if 'INJURED' in col and 'NUMBER' in col]
    killed_cols = [col for col in df.columns if 'KILLED' in col and 'NUMBER' in col]
    
    df["TOTAL_INJURED"] = df[injury_cols].sum(axis=1) if injury_cols else 0
    df["TOTAL_KILLED"] = df[killed_cols].sum(axis=1) if killed_cols else 0
    
    # Create database
    conn = sqlite3.connect("crashes.db")
    df.to_sql("crashes", conn, index=False, if_exists="replace")
    
    # Create indexes
    conn.execute("CREATE INDEX idx_year ON crashes(YEAR)")
    conn.execute("CREATE INDEX idx_borough ON crashes(BOROUGH)")
    
    print("Database created successfully!")
    conn.close()

if __name__ == "__main__":
    create_database()

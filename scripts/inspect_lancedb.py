import lancedb
import argparse
import os
import pandas as pd

def inspect(db_path, table_name="memories"):
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return

    db = lancedb.connect(db_path)
    try:
        tbl = db.open_table(table_name)
    except Exception as e:
        print(f"Error opening table {table_name}: {e}")
        return

    count = len(tbl)
    print(f"Table '{table_name}' has {count} rows.")
    
    if count > 0:
        print("\nSample row (excluding vector for brevity):")
        df = tbl.head(1).to_pandas()
        # Drop vector column for display
        if "vector" in df.columns:
            df = df.drop(columns=["vector"])
        print(df.iloc[0].to_dict())
        
        # Check vector dim
        vec = tbl.head(1).to_pandas().iloc[0]["vector"]
        print(f"\nVector dimension: {len(vec)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/lancedb_bge_small_full")
    args = parser.parse_args()
    inspect(args.db_path)

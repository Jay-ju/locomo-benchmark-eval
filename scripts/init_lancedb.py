import os
import argparse
import lancedb
import pyarrow as pa

def init_lancedb(db_path, vector_dim, schema_mode="basic"):
    db_path = os.path.expanduser(db_path)
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    table_name = "memories"

    print(f"Initializing LanceDB at {db_path} (dim={vector_dim}, mode={schema_mode})")
    
    # Define Schema (Same as in vector_store.py)
    fields = [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        pa.field("category", pa.string()),
        pa.field("importance", pa.float64()),
    ]
    
    if schema_mode == "basic":
        fields.append(pa.field("createdAt", pa.int64()))
    else:
        fields.append(pa.field("timestamp", pa.int64()))
        fields.append(pa.field("scope", pa.string()))
        fields.append(pa.field("metadata", pa.string()))
        
    schema = pa.schema(fields)

    try:
        # mode="create" -> raise if exists
        db.create_table(table_name, schema=schema, mode="create")
        print(f"Success: Table '{table_name}' created.")
    except Exception as e:
        print(f"Table '{table_name}' already exists or error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/lancedb_bge_small_full")
    parser.add_argument("--vector-dim", type=int, default=512)
    parser.add_argument("--schema-mode", default="basic")
    args = parser.parse_args()
    
    init_lancedb(args.db_path, args.vector_dim, args.schema_mode)

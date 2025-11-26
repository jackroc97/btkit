import databento as db
import duckdb
import glob
import sys

import polars as pl


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str):
    return conn.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
    ).fetchone()[0] > 0


def build_database(database_path: str, raw_data_folder: str):
    definition_paths = glob.glob(f"{raw_data_folder}/**/*.definition.dbn.zst", recursive=True)
    ohlcv_paths = glob.glob(f"{raw_data_folder}/**/*.ohlcv*.dbn.zst", recursive=True)

    print(f"Creating database from the following dbn files: {[*definition_paths, *ohlcv_paths]}")
    proceed = input("Would you like to proceed? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborting.")
        return
    
    print(f"Creating database {database_path}")
    
    # Connect to the database
    conn = duckdb.connect(database_path)
    
    # Load definition data
    print(f"Loading definitions from {len(definition_paths)} files...")
    for i, def_path in enumerate(definition_paths):
        print("Processing file:", def_path)
        parq_file = f"{raw_data_folder}/tmp_defn_{i}.parquet"
        definition_df = db.DBNStore.from_file(def_path).to_parquet()
        
        print("Converting to polars DataFrame and processing timestamps...")
        definition_df = pl.read_parquet(parq_file)
        definition_df.with_columns(
            pl.col("ts_event").dt.timestamp("us").alias("ts_event"),
            pl.col("ts_expiration").dt.timestamp("us").alias("ts_expiration"),
        )
        
        print("Inserting into database...")
        if table_exists(conn, "definition"):
            conn.execute(f"INSERT INTO definition SELECT * FROM definition_df")
        else:
            duckdb.sql("CREATE TABLE definition AS SELECT * from definition_df", connection=conn)

    # Load ohlcv data
    print(f"Loading OHLCV data from {len(ohlcv_paths)} files...")
    for i, ohlcv_path in enumerate(ohlcv_paths):
        print("Processing file:", ohlcv_path)
        parq_file = f"{raw_data_folder}/tmp_ohlcv_{i}.parquet"
        ohlcv_df = db.DBNStore.from_file(ohlcv_path).to_parquet()

        print("Converting to polars DataFrame and processing timestamps...")
        ohlcv_df = pl.read_parquet(parq_file)
        ohlcv_df = ohlcv_df.with_columns(
            pl.col("ts_event").dt.timestamp("us").alias("ts_event"),
        )

        #ohlcv_df["ts_event"] = ohlcv_df.index.astype(int) // 10**9
        #ohlcv_df = ohlcv_df[["ts_event", *cols]]

        if table_exists(conn, "ohlcv"):
            conn.execute(f"INSERT INTO ohlcv SELECT * FROM ohlcv_df")
        else:
            duckdb.sql("CREATE TABLE ohlcv AS SELECT * from ohlcv_df", connection=conn)
    
    print("Done!")


if __name__ == "__main__":
    build_database(*sys.argv[1:])
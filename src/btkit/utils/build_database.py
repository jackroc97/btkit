import databento as db
import duckdb
import glob
import polars as pl
import sys

from datetime import datetime


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str):
    return conn.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
    ).fetchone()[0] > 0


def build_database(raw_data_path: str, output_db_path: str):
    definition_files = sorted(glob.glob(f"{raw_data_path}/**/*.definition.*.dbn.zst"))
    ohlcv_files = sorted(glob.glob(f"{raw_data_path}/**/*.ohlcv-*.*.dbn.zst"))

    filenames = '\n'.join([*definition_files, *ohlcv_files])
    print(f"Building database from the following files:\n{filenames}")
    print(f"Output database path: {output_db_path}")
    proceed = input("Would you like to proceed? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborting.")
        return

    conn = duckdb.connect(output_db_path)

    print("Processing definition files...")
    for defn_file in definition_files:
        t0 = datetime.now()
        print(f"Processing {defn_file}...")

        # Read data from dbn format and convert to a Polars DataFrame
        print("Loading data from dbn...")
        df = pl.from_pandas(db.DBNStore.from_file(defn_file).to_df())

        # Remove user-defined instruments (e.g., spreads)
        df = df.filter(pl.col("user_defined_instrument") == 'N')

        # Convert datetime columns to unix microseconds
        df = df.with_columns([
            (pl.col("ts_event").dt.cast_time_unit("ms").cast(pl.Int64)).alias("ts_event_ms"),
            (pl.col("expiration").dt.cast_time_unit("ms").cast(pl.Int64)).alias("expiration_ms"),
            (pl.col("activation").dt.cast_time_unit("ms").cast(pl.Int64)).alias("activation_ms"),
        ])
        
        # Insert into duckdb in chunks
        chunk_size = 100_000
        for start in range(0, df.height, chunk_size):
            end = min(start + chunk_size, df.height)
            print(f"Inserting rows {start} to {end} into database...")
            defn_chunk = df[start:end]
            if table_exists(conn, "definition"):
                conn.execute(f"INSERT INTO definition SELECT * FROM defn_chunk")
            else:
                duckdb.sql("CREATE TABLE definition AS SELECT * from defn_chunk", connection=conn)
                
        print(f"Processed file in {(datetime.now() - t0).seconds} seconds")

    valid_instrument_ids = conn.execute("SELECT list(DISTINCT instrument_id) AS values FROM definition").fetchall()[0][0]

    print("Processing OHLCV files...")
    for ohlcv_file in ohlcv_files:
        t0 = datetime.now()
        print(f"Processing {ohlcv_file}...")

        # Read data from dbn format and convert to a Polars DataFrame
        print("Loading data from dbn...")
        df = pl.from_pandas(db.DBNStore.from_file(ohlcv_file).to_df().reset_index(names="ts_event"))

        # Filter to only valid instrument IDs
        df = df.filter(pl.col("instrument_id").is_in(valid_instrument_ids))

        # Convert datetime columns to unix microseconds
        df = df.with_columns([
            (pl.col("ts_event").dt.cast_time_unit("ms").cast(pl.Int64)).alias("ts_event_ms"),
        ])
        
        # Insert into duckdb in chunks
        chunk_size = 100_000
        for start in range(0, df.height, chunk_size):
            end = min(start + chunk_size, df.height)
            print(f"Inserting rows {start} to {end} into database...")
            ohlcv_chunk = df[start:end]
            if table_exists(conn, "ohlcv"):
                conn.execute(f"INSERT INTO ohlcv SELECT * FROM ohlcv_chunk")
            else:
                duckdb.sql("CREATE TABLE ohlcv AS SELECT * from ohlcv_chunk", connection=conn)
                
        print(f"Processed file in {(datetime.now() - t0).seconds} seconds")

    print("Done!")
    

if __name__ == "__main__":
    build_database(*sys.argv[1:])
    
import duckdb
import glob
import polars as pl
import sys


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str):
    return conn.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
    ).fetchone()[0] > 0
    

def combine_backtest_results(backtest_logs_path: str, output_db_path: str):
    conn = duckdb.connect(output_db_path)

    # Collect all output files
    metadata_files = glob.glob(f"{backtest_logs_path}/**/worker_*_metadata.json", recursive=True)
    trade_files = glob.glob(f"{backtest_logs_path}/**/worker_*_trade.parquet", recursive=True)

    schema = {
        "id": pl.Int64,
        "strategy_name": pl.Utf8,
        "strategy_version": pl.Utf8,
        "strategy_params": pl.Struct({}),
        "starting_cash": pl.Float64,
        "start_time": pl.Datetime,
        "end_time": pl.Datetime,
        "run_error": pl.Utf8
    }

    # Grab all metadata files and combine into a single dataframe
    metadata_df = pl.concat(pl.read_json(path, schema=schema) for path in metadata_files)
    conn.execute("CREATE OR REPLACE TABLE backtest AS SELECT * FROM metadata_df")

    # Read trades individually and put into database
    for path in trade_files:
        trade_df = pl.read_parquet(path)
        if table_exists(conn, "trade"):
            conn.execute("INSERT INTO trade SELECT * FROM trade_df")
        else:
            duckdb.sql("CREATE TABLE trade AS SELECT * from trade_df", connection=conn)
    
    conn.close()
    
if __name__ == "__main__":
    combine_backtest_results(*sys.argv[1:])


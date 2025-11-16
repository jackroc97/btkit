import itertools
import multiprocessing
import pandas as pd
import sqlite3
import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Type

from ..strategy import Strategy, DateSettings


class MatrixRunner:
    def __init__(self, strategy_type: Type[Strategy], matrix_path: str):
        self.strategy_type = strategy_type
        
        self.matrix_df = self.expand_matrix(matrix_path)
        
        
    def expand_matrix(self, matrix_path: str) -> pd.DataFrame:
        
        # Load the design matrix from yaml
        with open(matrix_path, "r") as f:
            param_space = yaml.safe_load(f)["param_space"]

        # Generate a design for each strategy type
        dfs = []
        all_columns = set()

        for strategy, params in param_space.items():
            keys, values = zip(*params.items())
            combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            df = pd.DataFrame(combos)
            dfs.append(df)
            all_columns.update(df.columns)

        # Normalize columns across all strategies 
        all_columns = sorted(all_columns)
        for i in range(len(dfs)):
            for col in all_columns:
                if col not in dfs[i].columns:
                    dfs[i][col] = None
            dfs[i] = dfs[i][all_columns]  # reorder columns

        # Combine all into a single matrix 
        matrix_df = pd.concat(dfs, ignore_index=True)

        # --- Add unique ID for tracking ---
        matrix_df.insert(0, "design_id", range(1, len(matrix_df) + 1))
        return matrix_df
        
        
    def run_series(self, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, output_db_path: str, date_settings: DateSettings = None):
        for _, row in tqdm(self.matrix_df.iterrows(), total=len(self.matrix_df)):
            strat = self.strategy_type(**row.to_dict())
            strat.run_backtest(starting_balance, start_time, end_time, time_step, output_db_path, date_settings, suppress=True)
       
    
    def _run_single_backtest(self, strategy_type: Type[Strategy], params: dict, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, output_db_path: str, date_settings: DateSettings):    
            strat = strategy_type(**params)
            strat.run_backtest(
                starting_balance,
                start_time,
                end_time,
                time_step,
                output_db_path,
                date_settings,
                suppress=True,
            )
            return True     
            
    def run_parallel(self, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, output_db_path: str, date_settings: DateSettings = None, max_workers: int = None):
        
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        tasks = []
        for _, row in self.matrix_df.iterrows():
            params = row.to_dict()
            tasks.append((
                self.strategy_type,
                params,
                starting_balance,
                start_time,
                end_time,
                time_step,
                output_db_path,
                date_settings,
            ))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._run_single_backtest, tasks), total=len(tasks)))
            
            
    def _resume_from(self, design_id: int, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, output_db_path: str, date_settings: DateSettings = None):
        self.matrix_df = self.matrix_df[self.matrix_df["design_id"] >= design_id]
        
        conn = sqlite3.connect(output_db_path)
        cur = conn.cursor()
        cur.execute(f"DELETE FROM trade WHERE session_id = {design_id};")
        cur.execute(f"DELETE FROM session WHERE id = {design_id};")
        conn.commit()
        conn.close()
        
        self.run(starting_balance, start_time, end_time, time_step, output_db_path, date_settings)

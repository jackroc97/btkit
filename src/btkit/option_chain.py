import pandas as pd
import sqlite3

from datetime import date, datetime, timedelta

from .instrument import InstrumentDetails


class OptionChain:
    now: datetime
    
    def __init__(self, instrument_details: InstrumentDetails, max_dte: int = 30, max_strike_dist: int = 100) -> None:
        self.instrument_details = instrument_details
        self.max_dte = max_dte
        self.max_strike_dist = max_strike_dist
        
        try:
            # Connect to the database and pull metadata for the underlying instrument
            conn = sqlite3.connect(self.instrument_details.data_db_path)
            cursor = conn.cursor()
            query = f"SELECT * FROM instrument WHERE symbol == '{self.instrument_details.symbol}' LIMIT 1"
            cursor.execute(query)
            instrument_info = cursor.fetchone()
            self.options_table = instrument_info[4]
        except sqlite3.Error as e:
            raise ValueError(f"No data found for symbol {instrument_details.symbol}")
        finally:
            if conn:
                conn.close()
    
    
    @classmethod
    def update_time(cls, now) -> None:
        cls.now = now
    
    
    def as_df(self) -> pd.DataFrame:
        start_datetime = int(OptionChain.now.timestamp())
        end_datetime = int((OptionChain.now + timedelta(days=self.max_dte)).timestamp())
        
        try:
            conn = sqlite3.connect(self.instrument_details.data_db_path)
    
            # TODO: I'd like to change the names of some of these columns to be more consistent
            query = f"""
                SELECT DISTINCT * FROM {self.options_table}
                WHERE 
                    quote_unixtime == {start_datetime} 
                    AND expire_unix >= {start_datetime}
                    AND expire_unix <= {end_datetime}
                    
            """
            
            # TODO: removed strike distance filter for now, since the database
            # is not indexed on that column and the query is very slow
            # AND strike_distance <= {self.max_strike_dist}
            return pd.read_sql_query(query, conn)
        except sqlite3.Error as e:
            raise ValueError(f"No data found for symbol {self.options_table}")
        finally:
            if conn:
                conn.close()
        
                
    def find_best_strike_by_delta(self, option_type: str, desired_exp: date, desired_delta: float) -> tuple[float, float]:
        chain = self.as_df()
        # TODO: May want to do this either once at the very start, or just make sure this is in the raw data
        chain["expiration"] = pd.to_datetime(chain["expire_unix"], unit='s').dt.strftime("%Y%m%d")
        expiration = desired_exp.strftime("%Y%m%d")
        matched_exp = chain[chain["expiration"] == expiration]
        if len(matched_exp) == 0:
            return None, None
    
        delta_col = f"{option_type.lower()}_delta"
        id_best_delta = (matched_exp[delta_col] - desired_delta).abs().idxmin()
        best_delta = matched_exp.loc[id_best_delta]
        best_strike = best_delta['strike']
        return best_strike, best_delta[delta_col]

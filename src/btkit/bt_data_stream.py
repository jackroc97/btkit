import pandas as pd
import sqlite3
import warnings

from datetime import datetime

from .bt_instrument_details import BtInstrumentDetails, BtInstrumentType


class BtDataStream:
    now: datetime

    def __init__(self, instrument_details: BtInstrumentDetails) -> None:
        self.instrument_details = instrument_details
         
        try:
            # Connect to the database and pull metadata for the underlying instrument
            conn = sqlite3.connect(self.instrument_details.data_db_path)
            cursor = conn.cursor()
            query = f"SELECT * FROM instrument WHERE symbol == '{self.instrument_details.symbol}' LIMIT 1"
            cursor.execute(query)
            instrument_info = cursor.fetchone()
            self.ohlcv_table = instrument_info[2]
            self.options_table = instrument_info[4]
            
            # If the instrument is an option, pull data from the options table
            # Otherwise, pull data from the ohlcv table
            if self.instrument_details.instrument_type == BtInstrumentType.OPTION:
                if not self.options_table:
                    raise ValueError(f"Could not find options table for {self.instrument_details.symbol}")
                exp_time = self.instrument_details.expiration_date.timestamp()
                query = f"""
                    SELECT DISTINCT * FROM {self.options_table}
                    WHERE 
                        expire_unix == {int(exp_time)}
                        AND strike == {int(self.instrument_details.strike)}
                """
                self._df = pd.read_sql_query(query, conn, index_col="quote_unixtime")
            elif self.ohlcv_table:
                query = f"SELECT * FROM {self.ohlcv_table}"
                self._df = pd.read_sql_query(query, conn, index_col="date")
            
            # Check for empty or duplicated data
            if self._df.empty:
                raise ValueError(f"No data found for symbol {self.instrument_details.symbol}")
            elif self._df.index.duplicated().any():
                self._df = self._df[~self._df.index.duplicated(keep='first')]
                warnings.warn(f"Data for symbol {self.instrument_details.symbol} contains duplicates. Duplicates have been removed.")
        except sqlite3.Error as e:
            raise ValueError(f"No data found for symbol {self.instrument_details.symbol}")
        finally:
            if conn:
                conn.close()
        
        
    @classmethod
    def update_time(cls, now) -> None:
        cls.now = now
        
        
    def exists(self, at_time: datetime = None, bars_ago: int = None) -> bool:
        at_time = at_time or BtDataStream.now
        if not bars_ago:
            return at_time.timestamp() in self._df.index
        else:
            # TODO: Implement
            raise NotImplementedError("Method exists with bars_ago parameter is not implemented.")
        
        
    def get(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or BtDataStream.now
        if self.exists(at_time=at_time, bars_ago=bars_ago):            
            if not bars_ago:
                return self._df.loc[at_time.timestamp(), name]
            else:
                # TODO: Implement
                raise NotImplementedError("Method get with bars_ago parameter is not implemented.")
        else:
            warnings.warn(f"Data for {name} at {at_time} does not exist in symbol {self.instrument_details.symbol}. Returning next available data.")
            return self._get_next(name, at_time=at_time, bars_ago=bars_ago)
        

    def _get_next(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or BtDataStream.now
        if not bars_ago:
            next_index = self._df.index[self._df.index > at_time.timestamp()]
            if not next_index.empty:
                return self._df.loc[next_index[0], name]
            else:
                raise ValueError(f"No data found for {name} after {at_time} on symbol {self.instrument_details.symbol}")
        else:
            # TODO: Implement
            raise NotImplementedError("Method _get_next with bars_ago parameter is not implemented.")


    def _get_last(self, name: str) -> any:
        return self._df[name].iloc[-1] if not self._df.empty else None

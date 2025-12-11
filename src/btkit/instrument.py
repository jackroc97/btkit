import duckdb
import numpy as np
import pandas as pd
import warnings

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum


def timestamp_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class OptionRight(Enum):
    CALL = "CALL"
    PUT = "PUT"
    
    
@dataclass
class Instrument:
    instrument_id: int
    symbol: str
    _df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    
    def __post_init__(self):
        self._df = InstrumentStore.ohlcv_for_instrument(self)
        
    
    def has_data(self) -> bool:
        return not self._df.empty
    
    
    def exists(self, at_time: datetime = None, bars_ago: int = None) -> bool:
        at_time = at_time or InstrumentStore.get_time()
        if not bars_ago:
            return timestamp_ms(at_time) in self._df.index
        else:
            # TODO: Implement
            raise NotImplementedError("Method exists with bars_ago parameter is not implemented.")
        
        
    def get(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or InstrumentStore.get_time()
        if self.exists(at_time=at_time, bars_ago=bars_ago):            
            if not bars_ago:
                return self._df.loc[timestamp_ms(at_time), name]
            else:
                # TODO: Implement
                raise NotImplementedError("Method get with bars_ago parameter is not implemented.")
        else:
            #warnings.warn(f"Data for {name} at {at_time} does not exist in symbol. Returning next available data.")
            return self.get_next(name, at_time=at_time, bars_ago=bars_ago)
        
        
    def get_next(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or InstrumentStore.get_time()
        if not bars_ago:
            next_index = self._df.index[self._df.index > timestamp_ms(at_time)]
            if not next_index.empty:
                return self._df.loc[next_index[0], name]
            else:
                return self._df.loc[self._df.index[-1], name]
                #raise ValueError(f"No data found for {name} after {at_time} on symbol {self.symbol}")
        else:
            # TODO: Implement
            raise NotImplementedError("Method get_next with bars_ago parameter is not implemented.")
        
    
    def __str__(self) -> str:
        return self.symbol
        

@dataclass
class Future(Instrument):
    expiration: datetime
    
    def search_options_by_delta(self, desired_delta: float, dte: int, 
                                option_right: OptionRight, **kwargs):
        return InstrumentStore.search_options_by_delta(self, desired_delta, dte, option_right, **kwargs)
    
    
    def get_option_chain(self, max_strike_dist: float = 200, max_dte: int = 15, time_tol: timedelta = timedelta(minutes=15)) -> pd.DataFrame: 
        # TODO: Do not use this in its present state... query needs to be updated to use pre-computed greeks
        # Query option chain for the given parameters
        #df = InstrumentStore.option_chain_for_instrument_id(self.instrument_id, max_strike_dist, max_dte, time_tol)
        #return df
        raise NotImplementedError("Not currently implemented, needs to be re-written to use pre-computed greeks")
    
    
@dataclass
class SpotFuture(Instrument):
    _exp_map: dict[datetime, int]
    _current_future: Future = None
    
    def __post_init__(self):
        pass
    
    def at(self, now: datetime) -> Future:
        exps = [exp for exp in self._exp_map.keys() if exp > now]
        if not exps:
            raise ValueError(f"No future found for symbol {self.symbol} at time {now}")
        
        fut_id = self._exp_map[min(exps)]
            
        if not self._current_future or self._current_future.instrument_id != fut_id:
            self._current_future = InstrumentStore.instrument_by_id(fut_id)
        return self._current_future
        
    
@dataclass
class Option(Instrument):
    expiration: datetime
    strike_price: float
    underlying_id: int
    multiplier: int
    right: OptionRight
    
    def __str__(self) -> str:
        symb = self.symbol.split(" ")[0:2]
        exp = self.expiration.strftime('%y%m%d')
        return f"{symb[0]} {exp}{symb[1]}"
        

class DbnInstrumentClass(Enum):
    CALL = "C"
    FUTURE = "F"
    PUT = "P"
    

import threading


_thread_local = threading.local()

class InstrumentStore:
    database_path: str = ""
    #_connection: duckdb.DuckDBPyConnection = None
    _now: datetime = None
    
    @staticmethod
    def connect_database(path: str) -> None:
        InstrumentStore.database_path = path
        #InstrumentStore._connection = duckdb.connect(database=InstrumentStore.database_path, read_only=True)
    
    @staticmethod
    def _get_connection() -> duckdb.DuckDBPyConnection:
        if not hasattr(_thread_local, "conn"):
            _thread_local.conn = duckdb.connect(InstrumentStore.database_path, read_only=True)
        return _thread_local.conn
        
    
    @staticmethod
    def set_time(now) -> None:
        _thread_local.now = now

    @staticmethod
    def get_time():
        return getattr(_thread_local, "now", None)
    

    @staticmethod
    def instrument_by_id(instrument_id: int, type_hint: type = None) -> 'Instrument':
        columns = ["instrument_id", "symbol", "expiration", "strike_price", "underlying_id", "unit_of_measure_qty", "instrument_class"]
        
        # This is necessary due to an annoying "feature" of databento's schema 
        # in which an instrument_id can be re-used on the same asset class
        if type_hint is Option:
            query = f"""
                SELECT {','.join(columns)}
                FROM definition
                WHERE instrument_id = {int(instrument_id)} 
                    AND instrument_class in ('C', 'P')              -- Only filter to only options
                    AND ts_event_ms <= {timestamp_ms(InstrumentStore.get_time())}       -- No options defined in the future 
                    AND expiration_ms >= {timestamp_ms(InstrumentStore.get_time())};    -- No options expiring in the past
            """
        else:
            query = f"""
                SELECT {','.join(columns)}
                FROM definition
                WHERE instrument_id = {int(instrument_id)}
            """
        result = InstrumentStore._get_connection().execute(query).fetchone()

        if result is None:
            raise ValueError(f"No instrument found for ID {instrument_id}")
        
        instrument_class = result[-1]
        if instrument_class == DbnInstrumentClass.FUTURE.value:
            return InstrumentStore.instrument_or_none(Future(*result[0:3]))
        elif instrument_class in {DbnInstrumentClass.CALL.value, DbnInstrumentClass.PUT.value}:
            right = OptionRight.CALL if instrument_class == DbnInstrumentClass.CALL.value else OptionRight.PUT
            return InstrumentStore.instrument_or_none(Option(*result[0:6], right))
        else:
            raise ValueError(f"Unsupported instrument class {instrument_class} for ID {instrument_id}")
    

    @staticmethod
    def future(symbol: str, expiration: date) -> 'Future':
        columns = ["instrument_id", "symbol", "expiration"]
        query = f"""
            SELECT {','.join(columns)}
            FROM definition
            WHERE instrument_class = 'F' AND symbol LIKE '{symbol}%' AND expiration::date = '{expiration.strftime("%Y-%m-%d")}';
        """
        result = InstrumentStore._get_connection().execute(query).fetchone()

        if result is None:
            warnings.warn(f"{InstrumentStore.get_time()} | No future found for symbol {symbol} expiring on {expiration}; returning None.")
            return None
        
        return InstrumentStore.instrument_or_none(Future(*result))
    

    @staticmethod
    def spot_future(symbol: str) -> 'SpotFuture':
        columns = ["instrument_id", "expiration"]
        query = f"""
            SELECT DISTINCT {','.join(columns)}
            FROM definition
            WHERE "group" = '{symbol}' AND instrument_class = 'F'
            ORDER BY expiration::date ASC;
        """
        result = InstrumentStore._get_connection().execute(query).fetchall()
        
        if result is None:
            raise ValueError(f"No futures found for group {symbol}")
        
        return SpotFuture(-1, symbol, _exp_map={ row[1]: row[0] for row in result })

    
    @staticmethod
    def option(underlying: 'Instrument', expiration: date, strike_price: float, right: OptionRight) -> Option:
        columns = ["instrument_id", "symbol", "expiration", "strike_price", "underlying_id", "unit_of_measure_qty", "instrument_class"]
        query = f"""
            SELECT {','.join(columns)}
            FROM definition
            WHERE underlying_id = {underlying.instrument_id}
              AND expiration::date = '{expiration.strftime("%Y-%m-%d")}'
              AND strike_price = {strike_price}
              AND instrument_class = '{right.value[0].capitalize()}';
        """
        result = InstrumentStore._get_connection().execute(query).fetchone()

        if result is None:
            warnings.warn(f"{InstrumentStore.get_time()} | No option found for underlying symbol {underlying.symbol} with parameters exp={expiration}, stk={strike_price}, right={right.value[0]}; returning None.")
            return None
        
        instrument_class = result[-1]
        right = OptionRight.CALL if instrument_class == DbnInstrumentClass.CALL.value else OptionRight.PUT
        return InstrumentStore.instrument_or_none(Option(*result[0:6], right))
        

    @staticmethod 
    def option_chain_for_instrument_id(underlying_id: int, max_strike_dist: float, max_dte: int, time_tol: timedelta) -> pd.DataFrame:
        tol_ms = time_tol.total_seconds() * 1000
        ts_unix_ms = timestamp_ms(InstrumentStore.get_time())        
        min_exp: datetime = InstrumentStore.get_time().replace(hour=0, minute=0, second=0, microsecond=0)
        max_exp: datetime = min_exp + timedelta(days=max_dte+1)

        # TODO: This is deprecated and should be replaced with a call to the option_greeks table
        query = f"""
            WITH
            -- Gather option ticks near current_dt
            option_data AS (
                SELECT 
                    o.instrument_id,
                    d.symbol,
                    d.expiration_ms,
                    d.expiration,
                    d.strike_price,
                    d.instrument_class,
                    o.open, o.high, o.low, o.close, o.volume,
                    o.ts_event_ms AS opt_ts
                FROM ohlcv o
                JOIN definition d ON o.instrument_id = d.instrument_id
                WHERE d.underlying_id = {underlying_id}
                  AND (d.expiration_ms BETWEEN {timestamp_ms(min_exp)} AND {timestamp_ms(max_exp)})
                  AND (o.ts_event_ms BETWEEN {ts_unix_ms - tol_ms} AND {ts_unix_ms + tol_ms})
            ),

            -- Gather underlying ticks near the same window
            underlying_data AS (
                SELECT 
                    ts_event_ms AS und_ts,
                    close AS und_close
                FROM ohlcv
                WHERE instrument_id = {underlying_id}
                  AND ts_event_ms BETWEEN {ts_unix_ms - tol_ms} AND {ts_unix_ms + tol_ms}
            ),

            -- For each option row, find the *closest* underlying tick
            matched AS (
                SELECT
                    od.opt_ts AS option_ts,
                    od.instrument_id,
                    od.symbol,
                    od.expiration_ms,
                    od.expiration,
                    od.strike_price,
                    od.instrument_class,
                    ud.und_close AS underlying_close,
                    od.open, od.high, od.low, od.close, od.volume,
                    ABS(od.opt_ts - ud.und_ts) AS time_diff
                FROM option_data od
                JOIN underlying_data ud
                ON od.opt_ts BETWEEN (ud.und_ts - {tol_ms}) AND (ud.und_ts + {tol_ms})
            ),

            -- Pick the closest underlying record per option tick
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY instrument_id ORDER BY time_diff ASC) AS rn
                FROM matched
            ),

            -- Apply filters (DTE and strike distance)
            filtered AS (
                SELECT
                    option_ts, instrument_id, symbol, expiration_ms, expiration, strike_price, instrument_class,
                    underlying_close, open, high, low, close, volume
                FROM ranked
                WHERE rn = 1
                  AND strike_price BETWEEN underlying_close - {max_strike_dist} AND underlying_close + {max_strike_dist}
            )
            SELECT * FROM filtered
            ORDER BY strike_price, instrument_class
        """
        conn = InstrumentStore._get_connection()
        rows = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]  # get column names
        df = pd.DataFrame(rows, columns=columns)
        #df = InstrumentStore._get_connection().execute(query).fetch_df()
        return df
    
    
    @staticmethod
    def ohlcv_for_instrument(instrument: Instrument) -> pd.DataFrame:
        time_filter = ""
        
        # NOTE: This is required due to the fact that instrument ids can be reused
        if isinstance(instrument, Option):
            ts_min = timestamp_ms(InstrumentStore.get_time())
            ts_max = timestamp_ms(instrument.expiration)
            time_filter = f"AND ts_event_ms BETWEEN {ts_min} AND {ts_max}"
        
        query = f"""
            SELECT ts_event_ms, open, high, low, close, volume
            FROM ohlcv
            WHERE instrument_id = {instrument.instrument_id} {time_filter}
            ORDER BY ts_event_ms ASC;
        """
        conn = InstrumentStore._get_connection()
        rows = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]  # get column names
        df = pd.DataFrame(rows, columns=columns)
        #df = InstrumentStore._get_connection().execute(query).fetch_df()
        if df.empty:
            # TODO: Print a warning here instead
            ...
            #raise ValueError(f"No ohlcv data found for ID {instrument.instrument_id} at time {InstrumentStore.get_time()}")
        
        df.set_index('ts_event_ms', inplace=True)
        return df


    @staticmethod
    def search_options_by_delta(underlying: Instrument, desired_delta: float, 
                                dte: int, option_right: OptionRight, at_time: datetime = None, 
                                time_tol: timedelta = timedelta(minutes=5), max_results: int = 10) -> pd.DataFrame: 
        if at_time is None:
            at_time = InstrumentStore.get_time()
        ts_min = timestamp_ms(at_time) - time_tol.total_seconds() * 1000
        ts_max = timestamp_ms(at_time) + time_tol.total_seconds() * 1000
        
        query = f"""
            SELECT 
                g.ts_event_ms,
                g.instrument_id,
                g.underlying_id,
                g.strike_price,
                g.delta,
                g.dte,
                ABS(g.delta - {desired_delta}) AS delta_diff
            FROM option_greeks g
            WHERE g.underlying_id = {underlying.instrument_id}
                AND g.ts_event_ms BETWEEN {ts_min} AND {ts_max}
                AND g.option_right = '{option_right.value.upper()[0]}'
                AND dte = {dte}
            ORDER BY delta_diff ASC
            LIMIT {max_results}
        """
        #return InstrumentStore._get_connection().execute(query).fetch_df()
        conn = InstrumentStore._get_connection()
        rows = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]  # get column names
        df = pd.DataFrame(rows, columns=columns)
        return df

    @staticmethod
    def instrument_or_none(instrument: Instrument) -> Instrument:
        if instrument.has_data():
            return instrument
        else: 
            warnings.warn(f"{InstrumentStore.get_time()} | {instrument} has no data; returning None.")
            return None
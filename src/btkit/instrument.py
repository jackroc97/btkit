import duckdb
import numpy as np
import pandas as pd
import warnings

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum

from .black76 import Black76


class OptionRight(Enum):
    CALL = "CALL"
    PUT = "PUT"
    
    
@dataclass
class Instrument:
    instrument_id: int
    symbol: str
    _df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    
    def __post_init__(self):
        self._df = InstrumentStore.ohlcv_for_instrument_id(self.instrument_id)
    
    
    def exists(self, at_time: datetime = None, bars_ago: int = None) -> bool:
        at_time = at_time or InstrumentStore.get_time()
        if not bars_ago:
            return at_time.timestamp() in self._df.index
        else:
            # TODO: Implement
            raise NotImplementedError("Method exists with bars_ago parameter is not implemented.")
        
        
    def get(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or InstrumentStore.get_time()
        if self.exists(at_time=at_time, bars_ago=bars_ago):            
            if not bars_ago:
                return self._df.loc[at_time.timestamp(), name]
            else:
                # TODO: Implement
                raise NotImplementedError("Method get with bars_ago parameter is not implemented.")
        else:
            #warnings.warn(f"Data for {name} at {at_time} does not exist in symbol. Returning next available data.")
            return self.get_next(name, at_time=at_time, bars_ago=bars_ago)
        
        
    def get_next(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or InstrumentStore.get_time()
        if not bars_ago:
            next_index = self._df.index[self._df.index > at_time.timestamp()]
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
    
    def get_option_chain(self, max_strike_dist: float = 200, max_dte: int = 15, time_tol: timedelta = timedelta(minutes=15)) -> pd.DataFrame: 
        # Query option chain for the given parameters
        df = InstrumentStore.option_chain_for_instrument_id(self.instrument_id, max_strike_dist, max_dte, time_tol)
        
        # Compute greeks
        # NOTE: If performance becomes a serious issue as more data is added, 
        # it may be worthwhile to pre-compute greeks
        df["T"] = (df["ts_expiration"] - df["option_ts"]) / (365.0 * 24 * 3600)
        df["sigma"] = df.apply(Black76.sigma, axis=1)
        df["delta"] = df.apply(Black76.delta, axis=1)
        return df
    
    
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
class ContinuousFuture(Instrument):
    instrument_ids: list[int]

    def __post_init__(self):
        self._df = InstrumentStore.continuous_ohlcv_for_instrument_ids(self.instrument_ids)
        
    
@dataclass
class Option(Instrument):
    expiration: datetime
    strike_price: float
    underlying_id: int
    multiplier: int
    right: OptionRight
        

class DbnInstrumentClass(Enum):
    CALL = "C"
    FUTURE = "F"
    PUT = "P"
    

class InstrumentStore:
    database_path: str = ""
    _connection: duckdb.DuckDBPyConnection = None
    _now: datetime = None
    
    @classmethod
    def connect_database(cls, path: str) -> None:
        cls.database_path = path
        cls._connection = duckdb.connect(database=cls.database_path, read_only=True)
    
    
    @classmethod
    def set_time(cls, now: datetime) -> None:
        cls._now = now
        
        
    @classmethod 
    def get_time(cls) -> datetime:
        return cls._now
    

    @classmethod
    def instrument_by_id(cls, instrument_id: int) -> 'Instrument':
        columns = ["instrument_id", "symbol", "expiration", "strike_price", "underlying_id", "unit_of_measure_qty", "instrument_class"]
        query = f"""
            SELECT {','.join(columns)}
            FROM definition
            WHERE instrument_id = {instrument_id};
        """
        result = cls._connection.execute(query).fetchone()

        if result is None:
            raise ValueError(f"No instrument found for ID {instrument_id}")
        
        instrument_class = result[-1]
        if instrument_class == DbnInstrumentClass.FUTURE.value:
            return Future(*result[0:3])
        elif instrument_class in {DbnInstrumentClass.CALL.value, DbnInstrumentClass.PUT.value}:
            right = OptionRight.CALL if instrument_class == DbnInstrumentClass.CALL.value else OptionRight.PUT
            return Option(*result[0:6], right)
        else:
            raise ValueError(f"Unsupported instrument class {instrument_class} for ID {instrument_id}")
    

    @classmethod
    def future(cls, symbol: str, expiration: date) -> 'Future':
        columns = ["instrument_id", "symbol", "expiration"]
        query = f"""
            SELECT {','.join(columns)}
            FROM definition
            WHERE instrument_class = 'F' AND symbol LIKE '{symbol}%' AND expiration::date = '{expiration.strftime("%Y-%m-%d")}';
        """
        result = cls._connection.execute(query).fetchone()

        if result is None:
            raise ValueError(f"No future found for symbol {symbol} expiring on {expiration}")
        
        return Future(*result)
    

    @classmethod
    def spot_future(cls, symbol: str) -> 'SpotFuture':
        columns = ["instrument_id", "expiration"]
        query = f"""
            SELECT DISTINCT {','.join(columns)}
            FROM definition
            WHERE "group" = '{symbol}' AND instrument_class = 'F'
            ORDER BY expiration::date ASC;
        """
        result = cls._connection.execute(query).fetchall()
        
        if result is None:
            raise ValueError(f"No futures found for group {symbol}")
        
        return SpotFuture(-1, symbol, _exp_map={ row[1]: row[0] for row in result })


    @classmethod 
    def continuous_future(cls, symbol: str) -> 'ContinuousFuture':
        columns = ["instrument_id", "symbol", "expiration"]
        query = f"""
            SELECT DISTINCT {','.join(columns)}
            FROM definition
            WHERE "group" = '{symbol}' AND instrument_class = 'F'
            ORDER BY expiration::date ASC;
        """
        result = cls._connection.execute(query).fetchall()
        
        if result is None:
            raise ValueError(f"No futures found for group {symbol}")
        
        return ContinuousFuture(instrument_id = -1, symbol=symbol, instrument_ids=[row[0] for row in result])
    
    
    @classmethod
    def option(cls, underlying: 'Instrument', expiration: date, strike_price: float, right: OptionRight) -> Option:
        columns = ["instrument_id", "symbol", "expiration", "strike_price", "underlying_id", "unit_of_measure_qty", "instrument_class"]
        query = f"""
            SELECT {','.join(columns)}
            FROM definition
            WHERE underlying_id = {underlying.instrument_id}
              AND expiration::date = '{expiration.strftime("%Y-%m-%d")}'
              AND strike_price = {strike_price}
              AND instrument_class = '{right.value[0].capitalize()}';
        """
        result = cls._connection.execute(query).fetchone()

        if result is None:
            raise ValueError(f"No option found for underlying ID {underlying.instrument_id}")
        
        instrument_class = result[-1]
        right = OptionRight.CALL if instrument_class == DbnInstrumentClass.CALL.value else OptionRight.PUT
        return Option(*result[0:6], right)
        

    @classmethod 
    def option_chain_for_instrument_id(cls, underlying_id: int, max_strike_dist: float, max_dte: int, time_tol: timedelta) -> pd.DataFrame:
        tol_seconds = time_tol.total_seconds()
        ts_unix = cls._now.timestamp()        
        min_exp: datetime = cls._now.replace(hour=0, minute=0, second=0, microsecond=0)
        max_exp: datetime = min_exp + timedelta(days=max_dte+1)

        query = f"""
            WITH
            -- Gather option ticks near current_dt
            option_data AS (
                SELECT 
                    o.instrument_id,
                    d.symbol,
                    d.ts_expiration,
                    d.expiration,
                    d.strike_price,
                    d.instrument_class,
                    o.open, o.high, o.low, o.close, o.volume,
                    o.ts_event AS opt_ts
                FROM ohlcv o
                JOIN definition d ON o.instrument_id = d.instrument_id
                WHERE d.underlying_id = {underlying_id}
                  AND (d.ts_expiration BETWEEN {min_exp.timestamp()} AND {max_exp.timestamp()})
                  AND (o.ts_event BETWEEN {ts_unix - tol_seconds} AND {ts_unix + tol_seconds})
            ),

            -- Gather underlying ticks near the same window
            underlying_data AS (
                SELECT 
                    ts_event AS und_ts,
                    close AS und_close
                FROM ohlcv
                WHERE instrument_id = {underlying_id}
                  AND ts_event BETWEEN {ts_unix - tol_seconds} AND {ts_unix + tol_seconds}
            ),

            -- For each option row, find the *closest* underlying tick
            matched AS (
                SELECT
                    od.opt_ts AS option_ts,
                    od.instrument_id,
                    od.symbol,
                    od.ts_expiration,
                    od.expiration,
                    od.strike_price,
                    od.instrument_class,
                    ud.und_close AS underlying_close,
                    od.open, od.high, od.low, od.close, od.volume,
                    ABS(od.opt_ts - ud.und_ts) AS time_diff
                FROM option_data od
                JOIN underlying_data ud
                ON od.opt_ts BETWEEN (ud.und_ts - {tol_seconds}) AND (ud.und_ts + {tol_seconds})
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
                    option_ts, instrument_id, symbol, ts_expiration, expiration, strike_price, instrument_class,
                    underlying_close, open, high, low, close, volume
                FROM ranked
                WHERE rn = 1
                  AND strike_price BETWEEN underlying_close - {max_strike_dist} AND underlying_close + {max_strike_dist}
            )
            SELECT * FROM filtered
            ORDER BY strike_price, instrument_class
        """
        df = cls._connection.execute(query).fetch_df()
        return df
        
    
    @classmethod
    def continuous_ohlcv_for_instrument_ids(cls, instrument_ids: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT 
                o.instrument_id,
                o.ts_event,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                d.expiration
            FROM ohlcv o
            JOIN definition d
                ON o.instrument_id = d.instrument_id
            WHERE o.instrument_id IN ({",".join(map(str, instrument_ids))})
        """
        df = cls._connection.execute(query).fetch_df()
        if df.empty:
            raise ValueError(f"No ohlcv data found for IDs {instrument_ids}")
        
        # Convert dates and sort
        df["expiration"] = pd.to_datetime(df["expiration"])
        df["ts_event"] = pd.to_datetime(df["ts_event"])
        df = df.sort_values(["expiration", "ts_event"]).reset_index(drop=True)

        # Determine contract order by expiration ---
        contracts = (
            df[["instrument_id", "expiration"]]
            .drop_duplicates()
            .sort_values("expiration")
            .reset_index(drop=True)
        )

        # Back-adjust each contract at rollover ---
        continuous = pd.DataFrame()
        adj_factor = 0.0  # running adjustment for continuity
        prev_close = None

        for i, row in contracts.iterrows():
            cid = row["instrument_id"]
            #exp = row["expiration"]

            sub = df[df["instrument_id"] == cid].copy()
            sub = sub.sort_values("ts_event")

            if i > 0:
                # Determine rollover date (last trading day of previous contract)
                prev_id = contracts.loc[i - 1, "instrument_id"]
                prev_sub = df[df["instrument_id"] == prev_id]
                last_prev = prev_sub["close"].iloc[-1]
                first_curr = sub["close"].iloc[0]

                # Compute ratio-based adjustment for continuity
                adj_factor += np.log(last_prev / first_curr)

            # Apply cumulative adjustment
            sub["adj_close"] = sub["close"] * np.exp(adj_factor)
            sub["adj_open"] = sub["open"] * np.exp(adj_factor)
            sub["adj_high"] = sub["high"] * np.exp(adj_factor)
            sub["adj_low"] = sub["low"] * np.exp(adj_factor)

            continuous = pd.concat([continuous, sub], ignore_index=True)

        # --- 5. Format and return ---
        continuous = continuous.sort_values("ts_event")
        continuous = continuous.set_index("ts_event")
        continuous = continuous[
            ["instrument_id", "expiration", "open", "high", "low", "close", 
             "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
        ]
        return continuous
    
    
    @classmethod
    def ohlcv_for_instrument_id(cls, instrument_id: int) -> pd.DataFrame:
        query = f"""
            SELECT ts_event, open, high, low, close, volume
            FROM ohlcv
            WHERE instrument_id = {instrument_id}
            ORDER BY ts_event ASC;
        """
        df = cls._connection.execute(query).fetch_df()
        if df.empty:
            raise ValueError(f"No ohlcv data found for ID {instrument_id}")
        
        df.set_index('ts_event', inplace=True)
        return df

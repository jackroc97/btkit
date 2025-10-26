import duckdb
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .black76 import Black76
from .data_stream import DataStream
    
    
class InstrumentType(Enum):
    STK = "STOCK"
    FUT = "FUTURE"
    CFT = "CONTINUOUS_FUTURE"
    OPT = "OPTION"
    FOP = "FUTURE_OPTION"
    
    
class OptionRight(Enum):
    CALL = "CALL"
    PUT = "PUT"
    
  
@dataclass
class Instrument:
    instrument_type: InstrumentType
    symbol: str
    base_symbol: str = None
    expiration_date: datetime = None
    underlying: 'Instrument' = None
    strike: float = None
    right: OptionRight = None
    multiplier: int = 1
    data: DataStream = None
    
        
    @classmethod
    def stock(cls, symbol: str) -> 'Instrument':
        return cls(InstrumentType.STK, symbol, data=DataStore.get_stock_or_future_data(symbol)) 
    
    
    @classmethod
    def future(cls, symbol: str, base_symbol: str, expiration_date: datetime) -> 'Instrument':
        return cls(InstrumentType.FUT, symbol, base_symbol, expiration_date, data=DataStore.get_stock_or_future_data(symbol))
    
    
    @classmethod
    def continuous_future(cls, symbol: str) -> 'Instrument':
        return cls(InstrumentType.CFT, symbol, symbol, data=DataStore.get_continuous_future_data(symbol))


    @classmethod
    def option(cls, underlying: 'Instrument', expiration_date: datetime, strike: float, right: OptionRight, multiplier: int = 100) -> 'Instrument':
        opt_type = InstrumentType.OPT if underlying.instrument_type != InstrumentType.FUT else InstrumentType.FOP
        symbol = DataStore.get_option_symbol(underlying, expiration_date, strike, right)
        opt = cls(opt_type, symbol, underlying.base_symbol, expiration_date=expiration_date, underlying=underlying, strike=strike, right=right, multiplier=multiplier)
        opt.data = DataStore.get_option_data(opt)
        return opt

    
    def get_option_chain(self, max_dte: int = 30, max_strike_dist: float = 100) -> pd.DataFrame:
        if self.instrument_type not in {InstrumentType.STK, InstrumentType.CFT, InstrumentType.FUT}:
            raise ValueError("Option chain can only be retrieved for stock or future instruments.")
        return DataStore.get_option_chain(self, max_dte, max_strike_dist)


class DataStore:
    now: datetime
    database_path: str = ""
    _connection: duckdb.DuckDBPyConnection = None
            
    @classmethod
    def connect_database(cls, path: str) -> None:
        cls.database_path = path
        cls._connection = duckdb.connect(database=cls.database_path, read_only=True)
            
            
    @classmethod
    def update_time(cls, now) -> None:
        cls.now = now
    
    
    # TODO: Implement
    @classmethod
    def get_stock_or_future_data(cls, symbol: str):
        raise NotImplementedError("Method _get_stock_or_future_data is not implemented.")
        
        
    @classmethod
    def get_continuous_future_data(cls, symbol: str):
        try:
            # Select nearest-expring quarterly contract from the database at each timestamp
            # TODO: Only get data when data is >= current time
            # TODO: Update the database schema to include an instrument_type column
            # so we can filter out non-future instruments
            df = cls._connection.execute(f"""
                WITH es_prices AS (
                    SELECT 
                        o.ts_unix,
                        o.close,
                        d.id AS instrument_id,
                        d.symbol,
                        d.expiration
                    FROM ohlcv o
                    JOIN definition d ON o.instrument_id = d.id
                    WHERE d.symbol LIKE '{symbol}%' AND LENGTH(d.symbol) = 4
                ),
                ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ts_unix
                               ORDER BY CAST(expiration AS TIMESTAMP) ASC
                           ) AS rn
                    FROM es_prices
                    WHERE CAST(expiration AS TIMESTAMP) > TO_TIMESTAMP(ts_unix)
                )
                SELECT ts_unix, close, instrument_id, symbol, expiration
                FROM ranked
                WHERE rn = 1
                ORDER BY ts_unix
            """).fetch_df()
            
        except duckdb.Error as e:
            raise ValueError(f"No data found for symbol {symbol.symbol}")
        
        # Convert timestamps and sort
        df['ts'] = pd.to_datetime(df['ts_unix'], unit='s')
        df['expiration'] = pd.to_datetime(df['expiration'])
        df.sort_values('ts', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Detect roll points (when instrument_id changes)
        df['contract_change'] = df['instrument_id'] != df['instrument_id'].shift()
        df['roll_ratio'] = 1.0  # default for non-roll points
        
        # TODO: Adjust open/high/low/volume as well
        # Compute roll ratios at roll points
        roll_indices = df.index[df['contract_change']].tolist()
        # Only compute ratio for roll indices from second roll onward
        for i in range(1, len(roll_indices)):
            prev_idx = roll_indices[i - 1]
            curr_idx = roll_indices[i]
            old_close = df.loc[prev_idx, 'close']
            new_close = df.loc[curr_idx, 'close']
            ratio = old_close / new_close if new_close != 0 else 1.0
            df.loc[curr_idx, 'roll_ratio'] = ratio
        
        # Apply cumulative back-adjustment (vectorized)
        df['adj_factor'] = df['roll_ratio'][::-1].cumprod()[::-1]  # propagate backward
        df['close'] = df['close'] * df['adj_factor']
        
        # Final cleanup
        df = df[['ts', 'close']]
        df.reset_index(drop=True, inplace=True)
        return df
    
    
    @classmethod 
    def get_option_symbol(cls, underlying: Instrument, expiration_date: datetime, strike: float, right: OptionRight) -> str:
        query = f"""
            SELECT symbol
            FROM definition
            WHERE base_symbol = {underlying.base_symbol}
              AND expiration = {expiration_date}
              AND strike = {strike}
              AND right = '{right.value[0].upper()}';
        """
        
        return cls._connection.execute(query).fetchone()
        
        
    @classmethod
    def get_option_data(cls, instrument: Instrument):
        try:
            # Find the instrument ID for the specified option
            option_def = cls._connection.execute(f"""
                SELECT id, symbol, expiration, strike, option_right
                FROM definition
                WHERE expiration = '{instrument.expiration_date}'
                    AND strike = {instrument.strike}
                    AND option_right = '{instrument.right.value[0].upper()}'
                LIMIT 1
            """).fetchdf()
            
            if option_def.empty:
                raise ValueError(f"No option found for {instrument.symbol} {instrument.expiration_date} {instrument.strike} {instrument.right.value}")
            else:
                option_id = option_def.iloc[0]["id"]
            
            # Query the OHLCV data for the option
            df = cls._connection.execute(f"""
                SELECT ts_unix, close AS option_close
                    FROM ohlcv
                    WHERE instrument_id = '{option_id}' AND ts_unix >= {int(DataStore.now.timestamp())}
                    ORDER BY ts_unix
            """).fetchdf()
            
        except duckdb.Error as e:
            raise ValueError(f"No option found for {instrument.symbol} {instrument.expiration_date} {instrument.strike} {instrument.right.value}")
        
        # Filter rows to return
        df = df[['ts', 'close']]
        
        # Convert timestamps and sort
        df['ts'] = pd.to_datetime(df['ts_unix'], unit='s')
        df.sort_values('ts', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
            
    @classmethod
    def get_option_chain(cls, instrument: Instrument, max_dte: int = 30, max_strike_dist: float = 100, time_tol: timedelta = timedelta(minutes=15)) -> pd.DataFrame:

        symbol = instrument.underlying.base_symbol if instrument.underlying else instrument.base_symbol
        underlying_last = instrument.data.get("close")
        tol_seconds = int(time_tol.total_seconds())

        # 1) SQL: find option defs that match underlying symbol prefix and strike/dte constraints,
        #    and pull the nearest ohlcv row within time tolerance for each option (using row_number partition)
        sql = f"""
        WITH candidates AS (
            SELECT
                d.id,
                d.symbol,
                d.expiration,
                d.strike,
                d.option_right,
                o.ts_unix,
                o.close AS option_close,
                ABS(o.ts_unix - {cls.now.timestamp()}) AS ts_diff
            FROM definition d
            JOIN ohlcv o ON o.instrument_id = d.id
            WHERE d.base_symbol = '{symbol}'
              AND d.strike IS NOT NULL
              AND ABS(o.ts_unix - {cls.now.timestamp()}) <= {tol_seconds}
              AND CAST(d.expiration AS DATE) - DATE('{cls.now.date()}') BETWEEN 0 AND {max_dte}
              AND ABS(d.strike - {underlying_last}) <= {max_strike_dist}
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts_diff ASC) AS rn
            FROM candidates
        )
        SELECT id, symbol, expiration, strike, option_right, ts_unix, option_close
        FROM ranked
        WHERE rn = 1
        ORDER BY strike, option_right
        """

        df = cls._connection.execute(sql).fetch_df()

        # close DB connection early
        cls._connection.close()

        if df.empty:
            # return empty dataframe with expected columns
            cols = ["id", "symbol", "expiration", "strike", "option_right",
                    "ts_unix", "option_ts", "option_close", "T", "implied_vol", "delta"]
            return pd.DataFrame(columns=cols)

        # 2) Post-process: convert types, compute T, filter sanity
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['option_ts'] = pd.to_datetime(df['ts_unix'], unit='s')
        # T in years (positive; clamp to small positive)
        df['T'] = (df['expiration'] - cls.now.date()).dt.total_seconds() / (365.0 * 24 * 3600)
        df['T'] = df['T'].clip(lower=1e-6)

        # Normalize option_right string
        df['option_right'] = df['option_right'].astype(str).str.lower().map(
            lambda s: 'call' if s.startswith('c') else ('put' if s.startswith('p') else s)
        )

        # 3) Compute implied vol and delta using warm starts.
        # We'll iterate through rows sorted by option_right then strike (so warm starts make sense)
        df.sort_values(['option_right', 'strike'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        implied_vols = []
        deltas = []
        prev_sigma = 0.4  # initial warm-start seed
        r = 0.01

        for idx, row in df.iterrows():
            opt_price = row['option_close']
            F = float(underlying_last)
            K = float(row['strike'])
            T = float(row['T'])
            opt_type = row['option_right']

            sigma = Black76.implied_vol(
                option_price=opt_price,
                F=F,
                K=K,
                T=T,
                r=r,
                option_type=opt_type,
                initial_guess=prev_sigma
            )

            implied_vols.append(sigma)

            if not np.isnan(sigma):
                delta = Black76.delta(F=F, K=K, T=T, r=r, sigma=sigma, option_type=opt_type)
                prev_sigma = sigma  # warm-start next row
            else:
                delta = np.nan

            deltas.append(delta)

        df['implied_vol'] = implied_vols
        df['delta'] = deltas

        # 4) Final column tidy
        result = df[[
            'id', 'symbol', 'expiration', 'strike', 'option_right',
            'ts_unix', 'option_ts', 'option_close', 'T', 'implied_vol', 'delta'
        ]].copy()

        # sort for readability: calls then puts, ascending strike
        result.sort_values(['option_right', 'strike'], inplace=True)
        result.reset_index(drop=True, inplace=True)

        return result
import pandas as pd
import warnings

from datetime import datetime


class DataStream:
    now: datetime

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        
    @classmethod
    def update_time(cls, now) -> None:
        cls.now = now
        
        
    def exists(self, at_time: datetime = None, bars_ago: int = None) -> bool:
        at_time = at_time or DataStream.now
        if not bars_ago:
            return at_time.timestamp() in self._df.index
        else:
            # TODO: Implement
            raise NotImplementedError("Method exists with bars_ago parameter is not implemented.")
        
        
    def get(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or DataStream.now
        if self.exists(at_time=at_time, bars_ago=bars_ago):            
            if not bars_ago:
                return self._df.loc[at_time.timestamp(), name]
            else:
                # TODO: Implement
                raise NotImplementedError("Method get with bars_ago parameter is not implemented.")
        else:
            warnings.warn(f"Data for {name} at {at_time} does not exist in symbol. Returning next available data.")
            return self._get_next(name, at_time=at_time, bars_ago=bars_ago)
        

    def _get_next(self, name: str, at_time: datetime = None, bars_ago: int = None) -> any:
        at_time = at_time or DataStream.now
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

from datetime import datetime

from .data_stream import DataStream
from .instrument_details import InstrumentDetails, OptionRight, InstrumentType
from .option_chain import OptionChain
    

class Instrument:
    
    @property
    def data_db_path(self) -> str:
        return self.instrument_details.data_db_path
    
    @property
    def instrument_type(self) -> InstrumentType:
        return self.instrument_details.instrument_type
    
    @property
    def symbol(self) -> str:
        return self.instrument_details.symbol
    
    @property
    def expiration_date(self) -> datetime:
        return self.instrument_details.expiration_date
    
    @property
    def strike(self) -> float:
        return self.instrument_details.strike
    
    @property
    def right(self) -> OptionRight:
        return self.instrument_details.right
    
    @property
    def multiplier(self) -> int:
        return self.instrument_details.multiplier
    
    
    def __init__(self, data_db_path: str, instrument_type: InstrumentType, symbol: str, expiration_date: datetime = None, strike: float = None, right: OptionRight = None, multiplier: int = 1):
        self.instrument_details = InstrumentDetails(data_db_path, instrument_type, symbol, expiration_date, strike, right, multiplier)
        self.data = DataStream(self.instrument_details)
        
    
    def init_options_chain(self, max_dte: int = 30, max_strike_dist: float = 100) -> None:
        self.options_chain = OptionChain(self.instrument_details, max_dte, max_strike_dist)


    def __str__(self):
        string = f"{self.symbol}"
        if self.instrument_type == InstrumentType.OPTION:
            string += f" {self.expiration_date.strftime('%Y%m%d')} {self.strike} {self.right.value}"
        return string
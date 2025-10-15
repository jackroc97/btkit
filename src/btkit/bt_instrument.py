from datetime import datetime

from .bt_data_stream import BtDataStream
from .bt_instrument_details import BtInstrumentDetails, BtOptionRight, BtInstrumentType
from .bt_options_chain import BtOptionsChain
    

class BtInstrument:
    
    @property
    def data_db_path(self) -> str:
        return self.instrument_details.data_db_path
    
    @property
    def instrument_type(self) -> BtInstrumentType:
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
    def right(self) -> BtOptionRight:
        return self.instrument_details.right
    
    @property
    def multiplier(self) -> int:
        return self.instrument_details.multiplier
    
    
    def __init__(self, data_db_path: str, instrument_type: BtInstrumentType, symbol: str, expiration_date: datetime = None, strike: float = None, right: BtOptionRight = None, multiplier: int = 1):
        self.instrument_details = BtInstrumentDetails(data_db_path, instrument_type, symbol, expiration_date, strike, right, multiplier)
        self.data = BtDataStream(self.instrument_details)
        
    
    def init_options_chain(self, max_dte: int = 30, max_strike_dist: float = 100) -> None:
        self.options_chain = BtOptionsChain(self.instrument_details, max_dte, max_strike_dist)


    def __str__(self):
        string = f"{self.symbol}"
        if self.instrument_type == BtInstrumentType.OPTION:
            string += f" {self.expiration_date.strftime('%Y%m%d')} {self.strike} {self.right.value}"
        return string
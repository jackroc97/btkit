from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class InstrumentType(Enum):
    STOCK = "STOCK"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    
    
class OptionRight(Enum):
    CALL = "CALL"
    PUT = "PUT"
    

@dataclass
class InstrumentDetails:
    data_db_path: str
    instrument_type: InstrumentType
    symbol: str
    expiration_date: datetime
    strike: float
    right: OptionRight
    multiplier: int = 1
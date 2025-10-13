from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class BtInstrumentType(Enum):
    STOCK = "STOCK"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    
    
class BtOptionRight(Enum):
    CALL = "CALL"
    PUT = "PUT"
    

@dataclass
class BtInstrumentDetails:
    data_db_path: str
    instrument_type: BtInstrumentType
    symbol: str
    expiration_date: datetime
    strike: float
    right: BtOptionRight
    
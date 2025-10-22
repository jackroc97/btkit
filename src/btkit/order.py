from dataclasses import dataclass
from enum import Enum

from .instrument import Instrument


class OrderAction(Enum):
    BTO = "BUY_TO_OPEN"
    STO = "SELL_TO_OPEN"
    BTC = "BUY_TO_CLOSE"
    STC = "SELL_TO_CLOSE"
    

@dataclass 
class Order:
    quantity: float
    instrument: Instrument
    
from dataclasses import dataclass
from enum import Enum

from .instrument import Instrument


class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    

@dataclass 
class Order:
    quantity: float
    instrument: Instrument
    
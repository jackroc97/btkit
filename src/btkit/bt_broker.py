from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .bt_instrument import BtInstrument
from .bt_instrument_details import BtInstrumentType


class BtOrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    

@dataclass 
class BtOrder:
    quantity: float
    instrument: BtInstrument
    

@dataclass
class BtPositionItem:
    quantity: float
    instrument: BtInstrument
    open_price: float = field(init=False)


    @property
    def market_price(self) -> float:
        price_col = "close"
        if self.instrument.instrument_type == BtInstrumentType.OPTION:
            price_col = f"{self.instrument.right.value[0].lower()}_last"
        return self.quantity * self.instrument.multiplier * self.instrument.data.get(price_col)
    
    
    @property
    def is_expired(self) -> bool:
        return self.instrument.data.now >= self.instrument.expiration_date 
    
    
    @property
    def pnl(self):
        return self.open_price - self.market_price
    
    
    def __post_init__(self):
        self.open_price = self.market_price
        
    
    def __str__(self):
        return f"{self.quantity}x {str(self.instrument)}"
    
    
@dataclass 
class BtPosition:
    items: list[BtPositionItem]
    open_side: BtOrderSide
    open_price: float = field(init=False)
    
    
    @property
    def market_price(self) -> float:
        return sum(item.market_price for item in self.items)
    
    
    @property
    def is_expired(self):
        return any(item.is_expired for item in self.items)
    
    
    @property
    def pnl(self):
        sign = -1 if self.open_side == BtOrderSide.SELL else 1
        return sum(sign * item.pnl for item in self.items)
    
    
    def __post_init__(self):
        sign = -1 if self.open_side == BtOrderSide.SELL else 1
        self.open_price = sign * sum(item.open_price for item in self.items)
    
    
    def __str__(self):
        return f"[{', '.join(str(item) for item in self.items)}]"


class BtBroker:
    
    def __init__(self, starting_cash: float):
        self.cash_balance = starting_cash
        self.positions: list[BtPosition] = []
        self._now: datetime = None
       
        
    def tick(self, now: datetime):
        self._now = now
        
        # Check if any positions are expired, and close them if so
        for position in self.positions:
            if position.is_expired:
                self._print_message(f"Found expired position: {position}")
                self.close_position(position)
        
        
    def open_position(self, side: BtOrderSide, *orders: BtOrder):
        position = BtPosition([BtPositionItem(o.quantity, o.instrument) for o in orders], side)
        
        if self.cash_balance + position.open_price > 0: 
            self.cash_balance += position.open_price
            self.positions.append(position)
            self._print_message(f"Opened new position: {position}")
            
        else:
            # TODO: Warn that position could not be opened!
            pass

        
    def close_position(self, position: BtPosition):
        self.cash_balance += position.market_price
        self.positions.remove(position)
        self._print_message(f"Closed position {position}")
            
    
    def _print_message(self, message):
        print(f"{self._now} | {message}")
        
from dataclasses import dataclass, field, _DefaultFactory
from uuid import uuid4

from .instrument import Instrument, InstrumentType
from .order import OrderAction


@dataclass
class PositionItem:
    quantity: float
    instrument: Instrument
    open_price: float = field(init=False)
    uuid: float = _DefaultFactory(uuid4)


    @property
    def market_price(self) -> float:
        price_col = "close"
        if self.instrument.instrument_type == InstrumentType.OPTION:
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
class Position:
    items: list[PositionItem]
    open_action: OrderAction
    open_price: float = field(init=False)
    uuid: float = _DefaultFactory(uuid4)
    
    
    @property
    def market_price(self) -> float:
        return sum(item.market_price for item in self.items)
    
    
    @property
    def is_expired(self):
        return any(item.is_expired for item in self.items)
    
    
    @property
    def pnl(self):
        sign = -1 if self.open_action == OrderAction.SELL else 1
        return sum(sign * item.pnl for item in self.items)
    
    
    def __post_init__(self):
        sign = -1 if self.open_action == OrderAction.SELL else 1
        self.open_price = sign * sum(item.open_price for item in self.items)
    
    
    def __str__(self):
        return f"[{', '.join(str(item) for item in self.items)}]"

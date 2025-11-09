from dataclasses import dataclass, field
from uuid import uuid4

from .instrument import Instrument, InstrumentStore, Future, Option
from .order import OrderAction


@dataclass
class PositionItem:
    quantity: float
    instrument: Instrument
    open_action: OrderAction
    open_price: float = field(init=False)
    uuid: float = field(default_factory=uuid4)


    @property
    def market_price(self) -> float:
        # TODO: If the instrument is an option that is at expiration,
        # compare against underlying to see if option is expiring worthless
        multiplier = 1
        if isinstance(self.instrument, Option):
            multiplier = self.instrument.multiplier
        #sign = 1 if self.open_action == OrderAction.STO else -1
        #sign = -1 if (self.open_action)
        close = self.instrument.get("close")
        return  abs(self.quantity) * multiplier * close
    
    
    @property
    def is_expired(self) -> bool:
        if isinstance(self.instrument, (Future, Option)):
            return InstrumentStore.get_time() >= self.instrument.expiration 
        return False
    
    
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
    open_price: float = field(init=False)
    uuid: float = field(default_factory=uuid4)
    
    
    @property
    def market_price(self) -> float:
        return sum(item.market_price for item in self.items)
    
    
    @property
    def is_expired(self):
        return any(item.is_expired for item in self.items)
    
    
    @property
    def pnl(self):
        return sum(item.pnl for item in self.items)
    
    
    @property
    def open_cash_effect(self):
        cash_eff = 0
        for item in self.items:
            cash_eff += self.market_price * -1 if item.open_action == OrderAction.BTO else 1
        return cash_eff
        
    
    @property
    def close_cash_effect(self):
        cash_eff = 0
        for item in self.items:
            cash_eff += self.market_price * -1 if item.open_action == OrderAction.BTC else 1
        return cash_eff
        
    
    def __post_init__(self):
        self.open_price = sum(item.open_price for item in self.items)
    
    
    def __str__(self):
        return f"[{', '.join(str(item) for item in self.items)}]"

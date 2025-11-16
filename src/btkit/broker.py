import warnings

from datetime import datetime

from .logger import Logger
from .order import Order, OrderAction
from .position import Position, PositionItem


class Broker:
    
    def __init__(self, starting_cash: float, logger: Logger, commission_per_contract: float = 0):
        self.cash_balance = starting_cash
        self.positions: list[Position] = []
        self.logger = logger
        self.commission_per_contract = commission_per_contract
        self._now: datetime = None
       
        
    def tick(self, now: datetime) -> None:
        self._now = now
        
        # Check if any positions are expired, and close them if so
        for position in self.positions:
            if position.is_expired:
                self.close_position(position)
        
        
    def open_position(self, *orders: Order) -> None:
        position = Position([PositionItem(o.quantity, o.instrument, o.action) for o in orders])
        
        commission_paid = sum([o.quantity for o in orders]) * self.commission_per_contract
        
        # The position open_price will be negative for a debit and positive for a credit
        # Commission is defined as positive and will always be applied to portfolio balance as a debit
        if self.cash_balance + position.open_price - commission_paid > 0: 
            self.cash_balance += (position.open_price - commission_paid)
            self.positions.append(position)
            self.logger.log_trade(self._now, position)
            
        else:
            warnings.warn(f"Could not open position at time {self._now} (balance = ${self.cash_balance})")
            pass

    
    def close_position(self, position: Position) -> None:
        self.cash_balance -= position.market_price
        self.positions.remove(position)
        self.logger.log_trade(self._now, position, is_closing=True)
        
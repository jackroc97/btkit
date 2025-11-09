from datetime import datetime

from .logger import Logger
from .order import Order, OrderAction
from .position import Position, PositionItem


class Broker:
    
    def __init__(self, starting_cash: float, logger: Logger):
        self.cash_balance = starting_cash
        self.positions: list[Position] = []
        self.logger = logger
        self._now: datetime = None
       
        
    def tick(self, now: datetime) -> None:
        self._now = now
        
        # Check if any positions are expired, and close them if so
        for position in self.positions:
            if position.is_expired:
                print(f"{self._now} | Found expired position: {position}")
                self.close_position(position)
        
        
    # TODO: Can we make OrderSide value either -1 SELL or 1 BUY and use that directly in the math...?
    def open_position(self, *orders: Order) -> None:
        position = Position([PositionItem(o.quantity, o.instrument, OrderAction.BTO if o.quantity > 0 else OrderAction.STO) for o in orders])
        cash_eff = position.open_cash_effect
        if self.cash_balance + cash_eff > 0: 
            self.cash_balance += cash_eff
            self.positions.append(position)
            self.logger.log_trade(self._now, position)
            print(f"{self._now} | Opened new position: {position}")
            
        else:
            # TODO: Warn that position could not be opened!
            pass

    
    def close_position(self, position: Position) -> None:
        cash_eff = position.close_cash_effect
        self.cash_balance += cash_eff
        self.positions.remove(position)
        self.logger.log_trade(self._now, position, is_closing=True)
        print(f"{self._now} | Closed position {position}")
        
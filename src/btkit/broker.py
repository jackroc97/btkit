from datetime import datetime
from enum import Enum

from .logger import Logger, TradeAction
from .order import Order, OrderAction
from .position import Position, PositionItem


class Broker:
    
    def __init__(self, starting_cash: float, log_db_path: str):
        self.cash_balance = starting_cash
        self.positions: list[Position] = []
        self.logger = Logger(log_db_path)
        self._now: datetime = None
       
        
    def tick(self, now: datetime) -> None:
        self._now = now
        
        # Check if any positions are expired, and close them if so
        for position in self.positions:
            if position.is_expired:
                print(f"{self._now} | Found expired position: {position}")
                self.close_position(position)
        
        
    # TODO: Can we make OrderSide value either -1 SELL or 1 BUY and use that directly in the math...?
    def open_position(self, side: OrderAction, *orders: Order) -> None:
        position = Position([PositionItem(o.quantity, o.instrument) for o in orders], side)
        
        if self.cash_balance + position.open_price > 0: 
            self.cash_balance += position.open_price
            self.positions.append(position)
            action = TradeAction.BTO if side == OrderAction.BUY else TradeAction.STO
            self.logger.log_trade(self._now, action, position)
            print(f"{self._now} | Opened new position: {position}")
            
        else:
            # TODO: Warn that position could not be opened!
            pass

    
    def close_position(self, position: Position) -> None:
        self.cash_balance += position.market_price
        self.positions.remove(position)
        action = TradeAction.STC if position.open_action == OrderAction.BUY else TradeAction.BTC
        self.logger.log_trade(self._now, action, position)
        print(f"{self._now} | Closed position {position}")
        
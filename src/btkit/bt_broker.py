from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Union

from .bt_instrument import BtInstrument


class BtOrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


class BtOrderType(Enum):
    MKT = "MKT"
    LMT = "LMT"
    STP = "STP"
    

@dataclass
class BtOrder:
    order_type: BtOrderType
    order_action: BtOrderAction
    quantity: float
    instrument: BtInstrument
    order_price: float = None
    
    
@dataclass
class BtComplexOrderLeg:
    order_action: BtOrderAction
    quantity: float
    instrument: BtInstrument
        
    
@dataclass 
class BtComplexOrder:
    order_type: BtOrderType
    legs: list[BtComplexOrderLeg]
    order_price: float = None
    
        
@dataclass
class BtPosition:
    quantity: float
    instrument: BtInstrument


class BtBroker:
    
    def __init__(self, starting_balance: float):
        self.cash_balance = starting_balance
        self.orders: list[Union[BtOrder, BtComplexOrder]] = []
        self.positions: list[BtPosition] = []
        
    
    def tick(self, now: datetime):
        
        # Check if any positions are expired, and close them if so
        for p in self.positions:
            if p.instrument.expiration_date and p.instrument.expiration_date >= now:
                close_action = BtOrderAction.SELL if p.quantity > 0 else BtOrderAction.BUY
                close_order = BtOrder(BtOrderType.MKT, close_action, p.quantity, p.instrument)
                self.orders.append(close_order)
        
        # Check if any orders can be filled
        for o in self.orders:
            self._try_fill_order(o)
                
    
    def place_order(self, order: Union[BtOrder, BtComplexOrder]): 
        self.orders.append(order)
        
            
    # TODO: Close price wont exist for options!
    def _get_mkr_price(self, order: Union[BtOrder, BtComplexOrder]) -> float:
        if isinstance(order, BtComplexOrder):
            mkt_price = 0
            for leg in order.legs:
                sign = -1 if leg.order_action == BtOrderAction.BUY else 1
                mkt_price += sign * leg.quantity * leg.instrument.data.get("close")
            return mkt_price
        else: 
            sign = -1 if order.order_action == BtOrderAction.BUY else 1
            return sign * order.quantity * order.instrument.data.get("close")
            
    
    def _fill_order(self, order: Union[BtOrder, BtComplexOrder], price: float):
        # Update the account cash balance
        self.cash_balance += price
        
        # Update the accout positions
        if isinstance(order, BtComplexOrder):
            for leg in order.legs:
                sign = -1 if leg.order_action == BtOrderAction.BUY else 1
                self.positions.append(BtPosition(sign * leg.quantity, leg.instrument))
                self.orders.remove(order)
        else:
            sign = -1 if order.order_action == BtOrderAction.BUY else 1
            self.positions.append(BtPosition(sign * order.quantity, order.instrument))
            self.orders.remove(order)
        
    
    def _try_fill_order(self, order: Union[BtOrder, BtComplexOrder]):
        mkt_price = self._get_mkr_price(order)
        
        if order.order_type == BtOrderType.MKT:
            # Fill the order at market price
            self._fill_order(order, mkt_price)
            
        elif order.order_type == BtOrderType.LMT:
            # If market price is negative then we are buying
            # Fill order if we can buy at or under the limit
            if mkt_price < 0 and mkt_price <= order.order_price:
                self._fill_order(order)      
                
            # If market price is positive then we are selling
            # Fill order if we can sell at or above the limit
            elif mkt_price > 0 and mkt_price >= order.order_price:
                self._fill_order(order)
            
        elif order.order_type == BtOrderType.STP:
            # If the market price is negative then we are buying
            # Fill the order if market price has surpassed the stop price
            if mkt_price < 0 and mkt_price >= order.order_price:
                self._fill_order(order)
                
            # If the market price is positive then we are selling
            # Fill the order if the market price as dopped below the stop price
            elif mkt_price > 0 and mkt_price <= order.order_price:
                self._fill_order(order)
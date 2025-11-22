import json
import polars as pl

from datetime import datetime
from pathlib import Path

from .instrument import Option
from .order import OrderAction
from .position import Position


class Logger:
    def __init__(self, strategy_name: str, strategy_version: str, strategy_params: dict, starting_balance: float, output_dir: str, worker_id: int = 1):
        self.worker_id = worker_id
        self.output_dir = Path(f"{output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=False)
    
        self.metadata = {
            "strategy_name": strategy_name,
            "strategy_version": strategy_version, 
            "strategy_params": strategy_params,
            "starting_cash": starting_balance,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.trade_rows = []


    def log_trade(self, time: datetime, postion: Position, is_closing: bool = False):
        self.trade_rows.extend([{
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "position_uuid": str(postion.uuid),
                "position_item_uuid": str(item.uuid),
                "action": str(item.open_action.value) if not is_closing else (str(OrderAction.BTC.value) if item.open_action == OrderAction.STO else str(OrderAction.STC.value)),
                "quantity": item.quantity,
                "mkt_price": item.market_price,
                "symbol": item.instrument.symbol,
                "expiration": item.instrument.expiration.strftime("%Y-%m-%d %H:%M:%S") if isinstance(item.instrument, Option) else None,
                "strike": item.instrument.strike_price if isinstance(item.instrument, Option) else None,
                "right_price": str(item.instrument.right.value) if isinstance(item.instrument, Option) else None,
                "multiplier": item.instrument.multiplier if isinstance(item.instrument, Option) else 1
            } for item in postion.items])


    def write_log(self):
        self.metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{self.output_dir}/worker_{self.worker_id}_metadata.json", "w") as metafile:
            metafile.write(json.dumps(self.metadata))

        if self.trade_rows:
            pl.DataFrame(self.trade_rows).write_parquet(
                self.output_dir / f"worker_{self.worker_id}_trade.parquet"
            )

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .broker import Broker
from .instrument import InstrumentStore
from .logger import Logger


@dataclass
class DateSettings:
    day_start: time = time(0, 0, 0)
    day_end: time = time(23, 59, 59, 999999)
    weekday_only: bool = False
    skip_dates: set[date] = field(default_factory=set)
    time_zone: ZoneInfo = ZoneInfo("America/New_York")
    

class Strategy:
    name: str
    version: str
    now: datetime
        
    def __init__(self, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, log_db_path: str, date_settings: DateSettings = None):
        self.date_settings = date_settings or DateSettings()   
        self.start_time = start_time.replace(tzinfo=date_settings.time_zone)
        self.end_time = end_time.replace(tzinfo=date_settings.time_zone)
        self.time_step = time_step
        self.now = self.start_time
        self.logger = Logger(log_db_path)  
        self.broker = Broker(starting_balance, self.logger)
        
    
    def run(self):
        t0 = datetime.now()
        self.logger.start_session(self.name, self.version)
        self.on_start()
        while self.now <= self.end_time:
            InstrumentStore.set_time(self.now)
            self.broker.tick(self.now)
            if self._should_tick():
                self.tick()
            self.now += self.time_step
        self.on_stop()
        self.logger.end_session()
        t1 = datetime.now()
        print(f"Backtest {self.logger.session_id} completed in {(t1-t0).total_seconds()} seconds!")
        
        
    def on_start(self):
        raise NotImplementedError("on_start method must be implemented by subclass.")
        
        
    def tick(self):
        raise NotImplementedError("tick method must be implemented by subclass.")
        
        
    def on_stop(self):
        raise NotImplementedError("on_stop method must be implemented by subclass.")
        
        
    def _should_tick(self) -> bool:
        return (self.now.time() >= self.date_settings.day_start and
                self.now.time() < self.date_settings.day_end and
                (self.now.weekday() < 5 if self.date_settings.weekday_only else True) and
                (self.now.date()) not in self.date_settings.skip_dates)
        
        
    def print_msg(self, msg: str) -> None:
        print(f"{self.now.strftime('%Y-%m-%d %H:%M:%S')} | {msg}") 
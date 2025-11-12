from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from tqdm import tqdm
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
                
    def __init__(self, **kwargs):
        self._params = kwargs
    
                
    def run_backtest(self, starting_balance: float, start_time: datetime, end_time: datetime, time_step: timedelta, output_db_path: str, date_settings: DateSettings = None, suppress: bool = False):
        # Configure backtest parameters
        self.date_settings = date_settings or DateSettings()   
        self.start_time = start_time.replace(tzinfo=date_settings.time_zone)
        self.end_time = end_time.replace(tzinfo=date_settings.time_zone)
        self.time_step = time_step
        self.now = self.start_time
        self.logger = Logger(output_db_path)  
        self.broker = Broker(starting_balance, self.logger)
        
        # Begin running the backtest
        t0 = datetime.now()
        self.logger.start_session(self.name, self.version, starting_balance, self._params)
        self.on_start()
        
        time_series = self._generate_time_series()
        for t in tqdm(time_series, total=len(time_series), disable=suppress):
            self.now = t
            InstrumentStore.set_time(self.now)
            self.broker.tick(self.now)
            self.tick()
            
        self.logger.end_session()
        t1 = datetime.now()
        
        tqdm.write(f"Backtest completed in {(t1-t0).total_seconds():.2f} seconds (session_id={self.logger.session_id})")
        
        
    def on_start(self):
        raise NotImplementedError("on_start method must be implemented by subclass.")
        
        
    def tick(self):
        raise NotImplementedError("tick method must be implemented by subclass.")
        
        
    def on_stop(self):
        raise NotImplementedError("on_stop method must be implemented by subclass.")
        
    
    def _generate_time_series(self):
        times = []
        current = self.start_time
        skip_dates = set(self.date_settings.skip_dates or [])

        while current <= self.end_time:
            date_only = current.date()
            time_only = current.time()

            # Apply filters
            if self.date_settings.weekday_only and current.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current += self.time_step
                continue
            if date_only in skip_dates:
                current += self.time_step
                continue
            if self.date_settings.day_start and time_only < self.date_settings.day_start:
                current += self.time_step
                continue
            if self.date_settings.day_end and time_only > self.date_settings.day_end:
                current += self.time_step
                continue

            times.append(current)
            current += self.time_step

        return times
        
    
    def write_message(self, msg: str) -> None:
        tqdm.write(f"{self.now.strftime('%Y-%m-%d %H:%M:%S')} | {msg}")
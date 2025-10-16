import os
import sqlite3

from datetime import datetime
from enum import Enum

from .position import Position


class TradeAction(Enum):
    BTO = "BUY_TO_OPEN"
    STO = "SELL_TO_OPEN"
    BTC = "BUY_TO_CLOSE"
    STC = "SELL_TO_CLOSE"


class Logger:
    
    def __init__(self, db_file_path: str):
        self.db_file_path = db_file_path
        
        if not os.path.exists(db_file_path):
            self.create_database(self.db_file_path)
        

    def start_session(self, strategy_name: str, strategy_version: str):
        self.con = sqlite3.connect(self.db_file_path)
        cur = self.con.cursor()
        
        cur.execute(f'''
            INSERT or IGNORE INTO strategy(name, version)
            VALUES ('{strategy_name}', '{strategy_version}')
        ''')
        self.con.commit()
            
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")

        cur.execute(f'''
            INSERT INTO session(strategy_name, strategy_version, start_time)
            VALUES('{strategy_name}', '{strategy_version}', '{start_time}')
        ''')
        self.con.commit()
        self.session_id = cur.lastrowid   
        
    
    def end_session(self):
        cur = self.con.cursor()
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
        cur.execute(f'''
            UPDATE session
            SET end_time = '{end_time}'
            WHERE id = {self.session_id}
        ''')
        self.con.commit()
        self.con.close()


    def log_trade(self, trade_time: datetime, trade_action: TradeAction, position: Position):
        cur = self.con.cursor() 
        for item in position.items:   
            cur.execute(f'''
                INSERT or IGNORE INTO trade(session_id, position_uuid, position_item_uuid, time, action, quantity, mkt_price, symbol, expiration, strike, right, multiplier)
                VALUES({self.session_id}, '{position.uuid}', '{item.uuid}', '{trade_time.strftime("%Y-%m-%d %H:%M:%S%z")}', '{trade_action}', {item.quantity}, {item.market_price}, '{item.instrument.symbol}', '{item.instrument.expiration_date}', {item.instrument.strike}, '{item.instrument.right.value}', {item.instrument.multiplier})
            ''')
        self.con.commit()
        return cur.lastrowid
        

    @classmethod
    def create_database(cls, db_file_path: str) -> None:
        if os.path.exists(db_file_path):
            print("Error: database already exists.")
            return
        
        con = sqlite3.connect(db_file_path)
        cur = con.cursor()
        
        create_strategy_table = f"""
            CREATE TABLE strategy(
                name    TEXT NOT NULL,
                version TEXT NOT NULL,
                PRIMARY KEY (name, version)
            )
            """
        cur.execute(create_strategy_table)

        create_session_table = f"""
            CREATE TABLE session(
                id                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                strategy_name       TEXT NOT NULL,
                strategy_version    TEXT NOT NULL,
                start_time          TEXT NOT NULL,
                end_time            TEXT,
                FOREIGN KEY(strategy_name, strategy_version) REFERENCES strategy(name, version)
            )
            """
        cur.execute(create_session_table)
        
        create_trades_table = f"""
            CREATE TABLE trade(
                id                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                session_id          INTEGER NOT NULL,
                position_uuid       TEXT NOT NULL,
                position_item_uuid  TEXT NOT NULL,
                time                TEXT NOT NULL,
                action              TEXT NOT NULL,
                quantity            REAL NOT NULL,
                mkt_price           REAL NOT NULL,
                symbol              TEXT NOT NULL,
                expiration          TEXT,
                strike              REAL,
                right               TEXT,
                multiplier          REAL,
            )
            """
        cur.execute(create_trades_table)
        
        con.commit()
        con.close()
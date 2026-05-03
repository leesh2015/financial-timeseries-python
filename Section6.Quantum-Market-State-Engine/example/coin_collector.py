import asyncio
import json
import websockets
import time
import os
from datetime import datetime, timezone
import pandas as pd
from redis.asyncio import Redis

class CoinCollector:
    def __init__(self, symbol="btcusdt", redis_url="redis://localhost"):
        self.symbol = symbol.lower()
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.ws_url = f"wss://fstream.binance.com/ws"
        
        self.trade_buffer = []
        self.jump_id = 0
        self.last_trade_ts = 0
        self.last_bid1 = 0.0
        self.last_ask1 = 0.0

        # === High-Performance Archiving ===
        self.save_buffer = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(base_dir), "data", self.symbol.upper())
        os.makedirs(self.data_dir, exist_ok=True)
        self.current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.is_flushing = False # Flag to prevent concurrent disk I/O

    async def collect(self):
        streams = [f"{self.symbol}@depth20@100ms", f"{self.symbol}@trade", f"{self.symbol}@aggTrade"]
        subscribe_msg = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        
        print(f"[*] Connecting Quantum Collector (Non-blocking I/O | Nano-Precision)...")
        list_key = f"quantum:final_jumps:theory:{self.symbol}"
        last_report_time = 0

        retry_delay = 1.0
        max_delay = 60.0

        while True:
            try:
                last_report_time = 0
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    retry_delay = 1.0 # Reset backoff on successful connect

                    async for message in ws:
                        data = json.loads(message)
                        e = data.get("e")
                        
                        if e == "depthUpdate":
                            bid1 = float(data["b"][0][0] if data.get("b") else 0)
                            ask1 = float(data["a"][0][0] if data.get("a") else 0)
                            
                            now = time.time()
                            if now - last_report_time > 1.0:
                                print(f"\r[*] {self.symbol.upper()} | B:{bid1:.1f} A:{ask1:.1f} | Saved:{self.jump_id}", end="", flush=True)
                                last_report_time = now

                            if bid1 != self.last_bid1 or ask1 != self.last_ask1:
                                if self.last_bid1 > 0:
                                    asyncio.create_task(self.publish_jump(list_key, data, bid1, ask1))
                                
                                self.last_bid1 = bid1
                                self.last_ask1 = ask1
                                
                        elif e in ["trade", "aggTrade"]:
                            ts = data.get("E", int(time.time() * 1000))
                            self.trade_buffer.append({
                                "offset_ms": ts - self.last_trade_ts if self.last_trade_ts > 0 else 1,
                                "volume": float(data["q"]),
                                "is_buy": not data.get("m", False),
                                "p": float(data["p"])
                            })
                            self.last_trade_ts = ts
            except Exception as e:
                print(f"\n[!] Connection lost: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay) # Exponential backoff

    async def publish_jump(self, list_key, depth, bid1, ask1):
        arrival_ns = time.time_ns() # Capture precise arrival in nanoseconds
        self.jump_id += 1
        impacts = self.trade_buffer
        self.trade_buffer = []
        
        jump_package = {
            "jump_id": self.jump_id,
            "bid1": bid1,
            "ask1": ask1,
            "impacts": impacts,
            "depth": {
                "bids": [[float(p), float(q)] for p, q in depth.get("b", [])],
                "asks": [[float(p), float(q)] for p, q in depth.get("a", [])]
            },
            "server_ts": depth.get("E", int(time.time() * 1000)),
            "arrival_ns": arrival_ns
        }

        # 1. Real-time Broadcast
        await self.redis.rpush(list_key, json.dumps(jump_package))
        await self.redis.ltrim(list_key, -100, -1)
        
        # 2. Archive to Buffer
        self.save_buffer.append({
            "jump_id": self.jump_id,
            "bid1": bid1,
            "ask1": ask1,
            "impacts": json.dumps(impacts),
            "depth": json.dumps(jump_package["depth"]),
            "server_ts": jump_package["server_ts"],
            "arrival_ns": arrival_ns
        })

        # 3. Non-blocking Flush (Threaded)
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if (now_date != self.current_date or len(self.save_buffer) >= 100) and not self.is_flushing:
            # Dispatch to thread pool to avoid blocking the event loop
            buffer_to_save = self.save_buffer[:]
            self.save_buffer = []
            target_date = self.current_date
            
            asyncio.create_task(self.async_flush(buffer_to_save, target_date))
            
            if now_date != self.current_date:
                self.current_date = now_date

    async def async_flush(self, data, date_str):
        self.is_flushing = True
        try:
            # Heavy pandas/disk operation moved to a thread
            await asyncio.to_thread(self.sync_flush, data, date_str)
        finally:
            self.is_flushing = False

    def sync_flush(self, data, date_str):
        file_path = os.path.join(self.data_dir, f"{date_str}.parquet")
        df_new = pd.DataFrame(data)
        
        try:
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['jump_id'])
            else:
                df_final = df_new
                
            df_final.to_parquet(file_path, index=False, engine='pyarrow')
            # Silent background log (optional)
        except Exception as e:
            print(f"\n[Disk Error] {e}")

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "btcusdt"
    collector = CoinCollector(symbol=symbol)
    try: asyncio.run(collector.collect())
    except KeyboardInterrupt: pass

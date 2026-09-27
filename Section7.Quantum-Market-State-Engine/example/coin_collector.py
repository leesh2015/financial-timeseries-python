import asyncio
import json
import websockets
import time
import os
import requests
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
        self.last_arrival_ns = time.perf_counter_ns()
        
        # === Local Order Book (Event-Time Management) ===
        self.bids = {} # {price: qty}
        self.asks = {}
        self.last_bid1 = 0.0
        self.last_ask1 = 0.0
        self.last_update_id = 0
        self.is_synced = False

        # === High-Performance Archiving ===
        self.save_buffer = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(base_dir), "data", self.symbol.upper())
        os.makedirs(self.data_dir, exist_ok=True)
        self.current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.is_flushing = False 
        self._last_perf_ns = time.perf_counter_ns()

    def get_initial_snapshot(self):
        """Fetch REST snapshot with better error handling and region fallback."""
        # Try both fapi.binance.com and binance.com as fallback
        endpoints = [
            f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=100",
            f"https://fapi.binance.me/fapi/v1/depth?symbol={self.symbol.upper()}&limit=100" # Alternative endpoint
        ]
        
        for url in endpoints:
            try:
                resp = requests.get(url, timeout=5)
                data = resp.json()
                
                if 'lastUpdateId' in data:
                    self.last_update_id = data['lastUpdateId']
                    self.bids = {float(p): float(q) for p, q in data['bids']}
                    self.asks = {float(p): float(q) for p, q in data['asks']}
                    self.update_best_quotes()
                    print(f"[*] Initial Snapshot Synced via {url} (UpdateID: {self.last_update_id})")
                    return True
                else:
                    print(f"[!] API Error via {url}: {data.get('msg', 'Unknown Error')} (Code: {data.get('code')})")
            except Exception as e:
                print(f"[!] Connection Error via {url}: {e}")
        
        print("[!] REST Snapshot failed. Entering 'Cold-Start' mode (Syncing via WebSocket only...)")
        return True # Continue anyway, will sync on first depthUpdate

    def update_best_quotes(self):
        self.last_bid1 = max(self.bids.keys()) if self.bids else 0.0
        self.last_ask1 = min(self.asks.keys()) if self.asks else 0.0

    async def collect(self):
        # 1. Start with REST Snapshot
        if not self.get_initial_snapshot():
            return

        # 2. Subscribe to REAL-TIME DIFF-DEPTH (Event-Time)
        # @depth is the diff-depth stream, which pushes every change as it happens.
        streams = [f"{self.symbol}@depth", f"{self.symbol}@trade", f"{self.symbol}@aggTrade"]
        subscribe_msg = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        
        print(f"[*] Connecting Quantum Collector (EVENT-TIME | Nano-Precision)...")
        list_key = f"quantum:final_jumps:theory:{self.symbol}"
        last_report_time = 0

        retry_delay = 1.0
        max_delay = 60.0

        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    retry_delay = 1.0 

                    async for message in ws:
                        data = json.loads(message)
                        e = data.get("e")
                        
                        if e == "depthUpdate":
                            await self.handle_depth_update(data, list_key)
                            
                            # Real-time dashboard feedback
                            now = time.perf_counter()
                            if now - last_report_time > 0.5:
                                print(f"\r[*] {self.symbol.upper()} | B:{self.last_bid1:.2f} A:{self.last_ask1:.2f} | Jumps:{self.jump_id} ", end="", flush=True)
                                last_report_time = now

                        elif e in ["trade", "aggTrade"]:
                            now_ns = time.perf_counter_ns()
                            offset_ns = now_ns - self._last_perf_ns
                            self.trade_buffer.append({
                                "offset_ms": int(offset_ns / 1000000),
                                "offset_ns": offset_ns,
                                "volume": float(data["q"]),
                                "is_buy": not data.get("m", False),
                                "p": float(data["p"])
                            })
                            self._last_perf_ns = now_ns

            except Exception as e:
                print(f"\n[!] Connection lost: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)

    async def handle_depth_update(self, data, list_key):
        """Apply diffs to local book and trigger Jump on Best Bid/Ask change."""
        u_final = data['u']
        
        # Standard Binance Sync Logic
        if u_final <= self.last_update_id:
            return # Old data
        
        # Update local book
        for p, q in data['b']:
            p, q = float(p), float(q)
            if q == 0: self.bids.pop(p, None)
            else: self.bids[p] = q
            
        for p, q in data['a']:
            p, q = float(p), float(q)
            if q == 0: self.asks.pop(p, None)
            else: self.asks[p] = q

        self.last_update_id = u_final
        
        # Check for Jump (Event-Time Trigger)
        new_bid1 = max(self.bids.keys()) if self.bids else 0.0
        new_ask1 = min(self.asks.keys()) if self.asks else 0.0
        
        if new_bid1 != self.last_bid1 or new_ask1 != self.last_ask1:
            # Prepare depth snapshot for engine (top 20 levels)
            depth_snap = {
                "b": sorted([[p, q] for p, q in self.bids.items()], key=lambda x: x[0], reverse=True)[:20],
                "a": sorted([[p, q] for p, q in self.asks.items()], key=lambda x: x[0])[:20]
            }
            
            # publish_jump is async
            await self.publish_jump(list_key, depth_snap, new_bid1, new_ask1, data.get("E", 0))
            
            self.last_bid1 = new_bid1
            self.last_ask1 = new_ask1

    async def publish_jump(self, list_key, depth, bid1, ask1, server_ts):
        arrival_ns = time.perf_counter_ns() 
        duration_ms = (arrival_ns - self.last_arrival_ns) / 1_000_000.0
        self.last_arrival_ns = arrival_ns
        
        self.jump_id += 1
        impacts = self.trade_buffer
        self.trade_buffer = []

        jump_package = {
            "jump_id": self.jump_id,
            "bid1": bid1,
            "ask1": ask1,
            "impacts": impacts,
            "depth": {
                "bids": depth["b"],
                "asks": depth["a"]
            },
            "server_ts": server_ts,
            "arrival_ns": arrival_ns,
            "duration_ms": duration_ms
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
            "server_ts": server_ts,
            "arrival_ns": arrival_ns,
            "duration_ms": duration_ms
        })

        # 3. Non-blocking Flush
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if (now_date != self.current_date or len(self.save_buffer) >= 500) and not self.is_flushing:
            buffer_to_save = self.save_buffer[:]
            self.save_buffer = []
            target_date = self.current_date
            asyncio.create_task(self.async_flush(buffer_to_save, target_date))
            if now_date != self.current_date:
                self.current_date = now_date

    async def async_flush(self, data, date_str):
        self.is_flushing = True
        try:
            await asyncio.to_thread(self.sync_flush, data, date_str)
        finally:
            self.is_flushing = False

    def sync_flush(self, data, date_str):
        dataset_path = os.path.join(self.data_dir, f"{date_str}.parquet")
        os.makedirs(dataset_path, exist_ok=True)
        chunk_filename = f"chunk_{int(time.time() * 1000)}.parquet"
        file_path = os.path.join(dataset_path, chunk_filename)
        df_new = pd.DataFrame(data)
        try:
            df_new.to_parquet(file_path, index=False, engine='pyarrow')
        except Exception as e:
            print(f"\n[Disk Error] {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Coin Collector v3.4.8 (Event-Time)")
    parser.add_argument('--symbol', type=str, default='btcusdt', help='Trading symbol')
    args = parser.parse_args()
    
    collector = CoinCollector(symbol=args.symbol.lower())
    try: 
        asyncio.run(collector.collect())
    except KeyboardInterrupt: 
        pass

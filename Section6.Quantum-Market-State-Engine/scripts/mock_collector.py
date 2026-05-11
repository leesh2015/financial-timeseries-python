import asyncio
import pandas as pd
import json
import logging
import os
import sys
import time
from redis.asyncio import Redis

# Add project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class MockCollector:
    """
    Quantum Mock Collector (v3.4.8 Simulation)
    Reads historical jump data and streams it to Redis to simulate a live market.
    """
    def __init__(self, data_path: str, symbol: str = "TQQQ", redis_url: str = "redis://localhost"):
        self.data_path = data_path
        self.symbol = symbol.upper()
        self.redis_url = redis_url
        self.redis = None
        self.df = None

    def load_data(self):
        logger.info(f"Loading sample data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            logger.error(f"Sample data not found at {self.data_path}. Please check the data directory.")
            sys.exit(1)
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df)} jump states. Ready to simulate.")

    async def run(self):
        self.load_data()
        self.redis = Redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        
        list_key = f"quantum:final_jumps:theory:{self.symbol}"
        logger.info(f"Mock Collector streaming to Redis: {list_key}")
        logger.info("Run 'quantum_predictor.py' in another terminal to see the results.")
        
        try:
            for i, row in self.df.iterrows():
                # 1. Parse impacts (Kinetic Energy sequence)
                impacts = []
                impact_raw = row.get('impact_json') or row.get('impacts')
                if impact_raw:
                    try:
                        if isinstance(impact_raw, str):
                            impacts = json.loads(impact_raw)
                        else:
                            impacts = list(impact_raw)
                    except:
                        pass

                # 2. Package the Jump Event
                payload = {
                    "date": str(row.get('date', '20260511')),
                    "jump_id": int(row['jump_id']),
                    "bid1": float(row['bid1']),
                    "ask1": float(row['ask1']),
                    "bid_vol1": float(row.get('bid_vol1', 0)),
                    "ask_vol1": float(row.get('ask_vol1', 0)),
                    "impacts": impacts,
                    "duration_ms": float(row.get('duration_ms', 100))
                }
                
                # 3. Stream to Redis (matching Live Predictor expectation)
                await self.redis.rpush(list_key, json.dumps(payload))
                await self.redis.ltrim(list_key, -100, -1)
                
                # 4. Simulate real-time interval (Accelerated simulation)
                # In a real market, jumps occur based on price changes.
                await asyncio.sleep(0.1) 
                
                if i % 100 == 0:
                    print(f"[*] Streamed {i}/{len(self.df)} jumps...")

            logger.info("Finished streaming all data.")

        except Exception as e:
            logger.error(f"Streaming Error: {e}")
        finally:
            await self.redis.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Mock Collector (Redis Sim)")
    parser.add_argument('--symbol', type=str, default='TQQQ', help='Symbol to simulate')
    parser.add_argument('--file', type=str, default='QQQ_jump_sample.parquet', help='Sample parquet file')
    args = parser.parse_args()
    
    data_file = os.path.join(project_root, 'data', args.file)
    collector = MockCollector(data_path=data_file, symbol=args.symbol)
    
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

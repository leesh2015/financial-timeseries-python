import asyncio
import pandas as pd
import json
import logging
import os
import sys

# Add project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class MockCollector:
    def __init__(self, data_path: str, host: str = '127.0.0.1', port: int = 9999):
        self.data_path = data_path
        self.host = host
        self.port = port
        self.df = None

    def load_data(self):
        logger.info(f"Loading sample data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            logger.error("Sample data not found. Please check the data directory.")
            sys.exit(1)
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df)} jump states. Ready to stream.")

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected: {addr}")
        
        try:
            for i, row in self.df.iterrows():
                # Construct payload identical to what Redis would broadcast
                impacts = []
                if 'impact_json' in row and row['impact_json']:
                    try:
                        impacts = json.loads(row['impact_json'])
                    except:
                        pass

                payload = {
                    "date": str(row.get('date', '20260424')),
                    "jump_id": row['jump_id'],
                    "bid1": row['bid1'],
                    "ask1": row['ask1'],
                    "bid_vol1": row['bid_vol1'],
                    "ask_vol1": row['ask_vol1'],
                    "impacts": impacts
                }
                
                # Send JSON string with newline terminator
                data = json.dumps(payload) + "\n"
                writer.write(data.encode('utf-8'))
                await writer.drain()
                
                # Simulate market delay (Accelerated time: 0.1s to 0.5s)
                import random
                await asyncio.sleep(random.uniform(0.05, 0.2))
                
            logger.info("Finished streaming all data.")
            writer.write(b'{"EOF": true}\n')
            await writer.drain()

        except ConnectionResetError:
            logger.warning("Client disconnected abruptly.")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Connection closed for {addr}")

    async def run(self):
        self.load_data()
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Mock Collector streaming server running on {addr}")
        logger.info("Run 'quantum_predictor.py' in another terminal to connect.")
        
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='QQQ_jump_sample.parquet', help='Sample parquet file name')
    args = parser.parse_args()
    
    data_file = os.path.join(project_root, 'data', args.file)
    collector = MockCollector(data_path=data_file)
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\nCollector stopped.")

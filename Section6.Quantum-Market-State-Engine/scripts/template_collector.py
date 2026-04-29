import asyncio
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BaseQuantumCollector(ABC):
    """
    Generalized Quantum Data Collector Template (Global Standard)
    
    This template implements the 'Lossless Accumulation' logic described in Lesson 2.
    Students should inherit this class and implement the broker-specific methods
    for connecting and parsing real-time data streams (e.g., Alpaca, Interactive Brokers, Robinhood).
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.jump_id = 0
        self.impact_sequence = []
        
        # Current state of the Order Book (Level 1)
        self.current_bid = 0.0
        self.current_ask = 0.0
        self.current_bid_vol = 0
        self.current_ask_vol = 0
        
    @abstractmethod
    async def connect_broker(self):
        """
        Connect to your broker's WebSocket or API.
        Must be implemented by the user.
        """
        pass

    @abstractmethod
    async def subscribe_market_data(self):
        """
        Subscribe to Level 1 Quotes (Bid/Ask) and Tick-by-tick trades.
        Must be implemented by the user.
        """
        pass

    def on_trade_received(self, volume: int, is_buy: bool, timestamp_ms: int):
        """
        [Lesson 2: Lossless Accumulation]
        Accumulate every single trade impact while the best bid/ask remains unchanged.
        This preserves the kinetic energy dynamics (volume and speed) without data loss.
        """
        impact = {
            "vol": volume,
            "buy": is_buy,
            "ms": timestamp_ms  # Time interval from the previous trade
        }
        self.impact_sequence.append(impact)

    def on_quote_received(self, bid: float, ask: float, bid_vol: int, ask_vol: int):
        """
        [Lesson 2: Quantum Jump Detection]
        Trigger a Quantum Jump event when the Best Bid or Best Ask changes.
        This signals that the potential barrier has been broken.
        """
        # If the bid or ask price changes, the potential barrier is broken
        if bid != self.current_bid or ask != self.current_ask:
            # Only finalize if we have a valid previous state (skip the very first initialization)
            if self.current_bid != 0.0 and self.current_ask != 0.0:
                self.finalize_quantum_jump(bid, ask, bid_vol, ask_vol)
            
            # Reset the state for the new potential barrier
            self.current_bid = bid
            self.current_ask = ask
            self.current_bid_vol = bid_vol
            self.current_ask_vol = ask_vol
            
            # Clear the accumulated impacts for the new jump
            self.impact_sequence = []
        else:
            # If price is the same, just update the volume (barrier thickness changes)
            self.current_bid_vol = bid_vol
            self.current_ask_vol = ask_vol

    def finalize_quantum_jump(self, new_bid: float, new_ask: float, new_bid_vol: int, new_ask_vol: int):
        """
        Package the accumulated impacts and the new quote state into a single jump event.
        """
        self.jump_id += 1
        
        # Package the data payload
        payload = {
            "date": datetime.now().strftime("%Y%m%d"),
            "jump_id": self.jump_id,
            "bid1": new_bid,
            "ask1": new_ask,
            "bid_vol1": new_bid_vol,
            "ask_vol1": new_ask_vol,
            "impacts": self.impact_sequence
        }
        
        # Log the event (optional)
        logger.info(f"Jump {self.jump_id} Confirmed: {len(self.impact_sequence)} impacts bundled.")
        
        # Broadcast or save the data
        self.broadcast_event(payload)

    def broadcast_event(self, payload: dict):
        """
        Send the packaged jump event to the Quantum Predictor.
        Users can implement Redis, TCP Sockets, ZeroMQ, or direct file saving here.
        """
        # Example: print JSON to stdout (can be piped to another process)
        # print(json.dumps(payload))
        pass

# ==========================================
# HOW TO USE (Example Implementation)
# ==========================================
# class MyBrokerCollector(BaseQuantumCollector):
#     async def connect_broker(self):
#         print("Connecting to Broker WebSocket...")
#         
#     async def subscribe_market_data(self):
#         print(f"Subscribing to {self.symbol}...")
#         
#     # In your WebSocket message handler:
#     async def on_message(self, message):
#         data = json.loads(message)
#         if data['type'] == 'trade':
#             self.on_trade_received(data['size'], data['side'] == 'buy', data['interval_ms'])
#         elif data['type'] == 'quote':
#             self.on_quote_received(data['bid'], data['ask'], data['bid_size'], data['ask_size'])

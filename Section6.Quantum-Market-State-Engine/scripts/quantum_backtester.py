import sys
import os
import pandas as pd
import json
import numpy as np
import logging

# Add project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.quantum_engine import QuantumDemonEngine, QuantumState
from core.quantum_models import calculate_physical_horizon

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class QuantumBacktester:
    def __init__(self, data_path: str, flip_mode: bool = False, base_threshold: float = 0.6):
        self.data_path = data_path
        self.engine = QuantumDemonEngine()
        self.tick_size = 0.01 # US Market Standard
        self.flip_mode = flip_mode # Data flip mode for bear market simulation
        self.base_threshold = base_threshold
        
    def run_simulation(self):
        logger.info(f"Loading data: {self.data_path} (Flip Mode: {self.flip_mode})")
        if not os.path.exists(self.data_path):
            logger.error("Data file not found.")
            return

        df = pd.read_parquet(self.data_path)
        total_rows = len(df)
        logger.info(f"Loaded a total of {total_rows} jump data points.")
        
        # Position and performance tracking variables
        active_positions = [] 
        completed_trades = []
        
        cumulative_pnl = 0.0
        wins = 0
        losses = 0
        
        print("="*80)
        print(" [ Quantum Demon Backtest Started - Harsh Mode ]")
        print("="*80)
        
        # Sequentially inject states to track engine state
        for i, row in df.iterrows():
            # 1. Check for previous position closures
            positions_to_close = [p for p in active_positions if i >= p['target_index']]
            active_positions = [p for p in active_positions if i < p['target_index']]
            
            for p in positions_to_close:
                direction = p['direction']
                entry_price = p['entry_price']
                
                # [Harsh Logic] Apply opposite order book price for closing (accept spread loss)
                # Close LONG by selling at Bid, Close SHORT by buying at Ask
                exit_price = row['bid1'] if direction == 1 else row['ask1']
                
                # Calculate PnL
                price_diff = exit_price - entry_price
                if self.flip_mode:
                    price_diff = -price_diff
                
                net_pnl = direction * price_diff 
                
                cumulative_pnl += net_pnl
                if net_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                    
                completed_trades.append({
                    'entry_idx': p['entry_index'],
                    'exit_idx': i,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_pnl': net_pnl
                })
            
            # 2. Load current State and inject to engine
            b_vol = row['buy_vol']
            s_vol = row['sell_vol']
            bv1 = row['bid_vol1']
            av1 = row['ask_vol1']
            
            if self.flip_mode:
                b_vol, s_vol = s_vol, b_vol # Flip trade volume
                bv1, av1 = av1, bv1         # Flip order book volume
                
            state = QuantumState(
                jump_id=row['jump_id'],
                bid1=row['bid1'],
                ask1=row['ask1'],
                bid_vol1=bv1,
                ask_vol1=av1,
                total_t=row['total_t'],
                buy_vol=b_vol,
                sell_vol=s_vol,
                duration_ms=row['duration_ms']
            )
            
            try:
                impact_seq = json.loads(row['impact_json']) if 'impact_json' in row else []
                if self.flip_mode:
                    for imp in impact_seq:
                        imp['side'] = 1 - imp['side']
            except:
                impact_seq = []
                
            # Engine processing
            result = self.engine.process_state(state, impact_seq)
            
            q_v = result['quantized_V']
            density = result['density']
            current_matrix = result['matrix']
            
            # --- [Physical Horizon Calculation] ---
            horizon_n = calculate_physical_horizon(current_matrix, target_gain=4.0)
            slippage_ticks = 1 if horizon_n > 200 else 3
                
            safe_v = max(1.0, q_v)
            dynamic_threshold = self.base_threshold + (0.2 / safe_v)
            
            m_future = np.linalg.matrix_power(current_matrix, horizon_n)
            prob_dist = m_future[2] 
            
            up_prob = prob_dist[0] + prob_dist[1]
            down_prob = prob_dist[3] + prob_dist[4]
            
            # [Harsh Logic] Determine entry price (Use actual fill price instead of Mid + apply slippage)
            # LONG entry: Buy at Ask1, SHORT entry: Sell at Bid1
            if self.flip_mode:
                if down_prob > dynamic_threshold:
                    # SHORT entry in mirror world
                    entry_price = state.bid1 - (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': i,
                        'target_index': i + horizon_n,
                        'direction': -1,
                        'entry_price': entry_price,
                        'slippage': slippage_ticks
                    })
                elif up_prob > dynamic_threshold:
                    # LONG entry in mirror world
                    entry_price = state.ask1 + (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': i,
                        'target_index': i + horizon_n,
                        'direction': 1,
                        'entry_price': entry_price,
                        'slippage': slippage_ticks
                    })
            else:
                if up_prob > dynamic_threshold:
                    # LONG entry: Buy at Ask1 and add slippage
                    entry_price = state.ask1 + (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': i,
                        'target_index': i + horizon_n,
                        'direction': 1,
                        'entry_price': entry_price,
                        'slippage': slippage_ticks
                    })
                elif down_prob > dynamic_threshold:
                    # SHORT entry: Sell at Bid1 and subtract slippage
                    entry_price = state.bid1 - (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': i,
                        'target_index': i + horizon_n,
                        'direction': -1,
                        'entry_price': entry_price,
                        'slippage': slippage_ticks
                    })

        # Final Report Output
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n============================================================")
        print(" [ Quantum Backtest Final Report ]")
        print("============================================================")
        print(f"Total Evaluated Jumps: {total_rows:,}")
        print(f"Total Executed Trades: {len(completed_trades):,}")
        print(f"Wins: {wins:,}")
        print(f"Losses: {losses:,}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Cumulative Net PnL: {cumulative_pnl:.2f} points")
        print("============================================================")
        if cumulative_pnl > 0:
            print("Conclusion: [SUCCESS] The engine preserved mathematical profit even under harsh trading environments.")
        else:
            print("Conclusion: [FAIL] Failed to overcome spread and slippage penalties. Logic tuning required.")
        print("============================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip', action='store_true', help='Data flip (bear market simulation) mode')
    parser.add_argument('--file', type=str, help='Specific file name or path to backtest')
    parser.add_argument('--threshold', type=float, default=0.6, help='Base threshold for entry confidence')
    args = parser.parse_args()
    
    # Data file path determination logic
    if args.file:
        if os.path.isabs(args.file) or os.path.dirname(args.file):
            data_file = args.file
        else:
            data_file = os.path.join(project_root, 'data', args.file)
    else:
        # Default: sample data in the repository
        data_file = os.path.join(project_root, 'data', 'QQQ_jump_sample.parquet')
        
    backtester = QuantumBacktester(data_file, flip_mode=args.flip, base_threshold=args.threshold)
    backtester.run_simulation()

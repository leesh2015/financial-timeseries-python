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
    def __init__(self, data_path: str, flip_mode: bool = False, 
                 base_threshold: float = 0.83, target_gain: float = 10.15,
                 vol_multiplier: float = 1.0, warmup_jumps: int = 100):
        self.data_path = data_path
        self.tick_size = 0.01 
        self.flip_mode = flip_mode 
        self.base_threshold = base_threshold
        self.target_gain = target_gain
        self.vol_multiplier = vol_multiplier
        self.warmup_jumps = warmup_jumps
        
        self.engine = QuantumDemonEngine(vol_multiplier=self.vol_multiplier)
        
    def run_simulation(self):
        logger.info(f"Loading data: {self.data_path} (Flip Mode: {self.flip_mode})")
        if not os.path.exists(self.data_path):
            logger.error("Data file not found.")
            return

        df = pd.read_parquet(self.data_path)
        total_rows = len(df)
        logger.info(f"Loaded a total of {total_rows} jump data points.")
        
        # Position and performance tracking
        active_positions = [] 
        completed_trades = []
        
        cumulative_pnl = 0.0
        wins = 0
        losses = 0
        
        print("="*80)
        print(f" [ Money Maker Hybrid Engine v2.3 - Theoretical Backtest ]")
        print(f" [ G:{self.target_gain} T:{self.base_threshold} V:{self.vol_multiplier} W:{self.warmup_jumps} ]")
        print("="*80)
        
        for i, row in df.iterrows():
            # 1. Position Closure Logic
            positions_to_close = [p for p in active_positions if i >= p['target_index']]
            active_positions = [p for p in active_positions if i < p['target_index']]
            
            for p in positions_to_close:
                direction = p['direction']
                entry_price = p['entry_price']
                exit_price = row['bid1'] if direction == 1 else row['ask1']
                
                price_diff = exit_price - entry_price
                if self.flip_mode: price_diff = -price_diff
                
                net_pnl = direction * price_diff 
                cumulative_pnl += net_pnl
                if net_pnl > 0: wins += 1
        # Progress bar
        total_rows = len(df)
        for idx, (i, state_row) in enumerate(df.iterrows()):
            if idx % 500 == 0:
                print(f"Progress: {idx}/{total_rows} jumps processed...")

            # Parse volumes with safety
            bv1 = state_row.get('bid_vol1') or state_row.get('bid_v1') or 0
            av1 = state_row.get('ask_vol1') or state_row.get('ask_v1') or 0
            b_vol = state_row.get('buy_vol') or 0
            s_vol = state_row.get('sell_vol') or 0

            state = QuantumState(
                jump_id=state_row['jump_id'], bid1=state_row['bid1'], ask1=state_row['ask1'],
                bid_vol1=bv1, ask_vol1=av1,
                total_t=state_row['total_t'], buy_vol=b_vol, sell_vol=s_vol,
                duration_ms=state_row['duration_ms']
            )
            
            # 3. Kinetic Sequence Reconstruction
            try:
                impact_seq = json.loads(state_row['impact_json']) if 'impact_json' in state_row else []
                if self.flip_mode:
                    for imp in impact_seq: imp['side'] = 1 - imp['side']
            except:
                impact_seq = []
            
            # --- 1. [v2.3 Sync] Zero Lag Matrix Update ---
            # Process with N=1 to get the CURRENT matrix and V
            res = self.engine.process_state(state, impact_seq, horizon_n=1.0)
            current_matrix = res['matrix']
            q_v = res.get('quantized_V', 1.0)
            
            # --- 2. [v2.3 Sync] Dynamic Horizon N ---
            horizon_n = calculate_physical_horizon(current_matrix, target_gain=self.target_gain)
            
            # --- 3. Close Positions ---
            positions_to_close = [p for p in active_positions if idx >= p['target_index']]
            active_positions = [p for p in active_positions if idx < p['target_index']]
            
            for p in positions_to_close:
                direction = p['direction']
                exit_price = state.bid1 if direction == 1 else state.ask1
                
                price_diff = exit_price - p['entry_price']
                if self.flip_mode: price_diff = -price_diff
                
                net_pnl = direction * price_diff
                cumulative_pnl += net_pnl
                if net_pnl > 0: wins += 1
                else: losses += 1
                
                completed_trades.append({
                    'entry_idx': p['entry_index'], 'exit_idx': idx,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': p['entry_price'], 'exit_price': exit_price, 'net_pnl': net_pnl
                })

            # --- 4. [v2.3 Sync] Symmetric Probability Split & Dynamic Threshold ---
            # Power the matrix to the calculated horizon N
            m_future = self.engine.matrix_builder.get_n_step_matrix(horizon_n)
            dim = res['dimension']
            mid = dim // 2
            prob_dist = m_future[mid] 
            
            if dim % 2 == 0:
                up_prob = np.sum(prob_dist[:mid])
                down_prob = np.sum(prob_dist[mid:])
            else:
                up_prob = np.sum(prob_dist[:mid])
                down_prob = np.sum(prob_dist[mid+1:])
            
            dynamic_threshold = self.base_threshold + (0.2 / max(1.0, q_v))
            is_warmup = idx < self.warmup_jumps
            
            if not is_warmup:
                slippage_ticks = 1 if horizon_n > 200 else 3
                
                if up_prob > dynamic_threshold:
                    entry_price = state.ask1 + (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': idx, 'target_index': idx + int(horizon_n),
                        'direction': 1, 'entry_price': entry_price
                    })
                elif down_prob > dynamic_threshold:
                    entry_price = state.bid1 - (slippage_ticks * self.tick_size)
                    active_positions.append({
                        'entry_index': idx, 'target_index': idx + int(horizon_n),
                        'direction': -1, 'entry_price': entry_price
                    })

        # [v2.3 Sync] Final Liquidation for accurate trade counting
        if active_positions:
            last_row = df.iloc[-1]
            for p in active_positions:
                direction = p['direction']
                exit_price = last_row['bid1'] if direction == 1 else last_row['ask1']
                
                price_diff = exit_price - p['entry_price']
                if self.flip_mode: price_diff = -price_diff
                
                net_pnl = direction * price_diff
                cumulative_pnl += net_pnl
                if net_pnl > 0: wins += 1
                else: losses += 1
                
                completed_trades.append({
                    'entry_idx': p['entry_index'], 'exit_idx': total_rows-1,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': p['entry_price'], 'exit_price': exit_price, 'net_pnl': net_pnl
                })

        # Final Report
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n" + "="*80)
        print(f" [ Quantum Backtest Final Report - Money Maker v2.3 ]")
        print("="*80)
        print(f"Total evaluated Jumps: {total_rows:,}")
        print(f"Total Executed Trades: {total_trades:,}") # <--- 정확한 집계로 수정
        print(f"Wins: {wins:,} / Losses: {losses:,}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Cumulative Net PnL: {cumulative_pnl:+.2f} points")
        print("="*80)
        
        if cumulative_pnl > 0:
            print("Conclusion: [SUCCESS] The engine preserved mathematical profit.")
        else:
            print("Conclusion: [FAIL] Failed to overcome friction. Parameter tuning required.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='QQQ_jump_sample.parquet')
    parser.add_argument('--threshold', type=float, default=0.83)
    parser.add_argument('--gain', type=float, default=10.15)
    parser.add_argument('--vol-multiplier', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()
    
    data_file = args.file
    if not os.path.isabs(data_file) and not os.path.dirname(data_file):
        data_file = os.path.join(project_root, 'data', args.file)
        
    backtester = QuantumBacktester(
        data_file, flip_mode=args.flip, 
        base_threshold=args.threshold, target_gain=args.gain,
        vol_multiplier=args.vol_multiplier, warmup_jumps=args.warmup
    )
    backtester.run_simulation()

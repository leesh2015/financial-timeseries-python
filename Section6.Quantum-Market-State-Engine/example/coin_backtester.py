import sys
import os
import pandas as pd
import json
import numpy as np
import logging
from datetime import datetime, timedelta

# Project Root Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from coin_engine import CoinEngine
from coin_models import CoinJumpState, KineticImpact, PotentialEnergyLevel, MarketPotential, calculate_physical_horizon

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CoinBacktester:
    def __init__(self, symbol: str, base_threshold: float = 0.83, target_gain: float = 6.0, 
                 vol_multiplier: float = 1.0, data_dir: str = None, velocity_only: bool = False):
        self.symbol = symbol.upper()
        self.engine = CoinEngine(vol_multiplier=vol_multiplier)
        self.tick_size = 0.01 
        self.base_threshold = base_threshold
        self.target_gain = target_gain
        self.velocity_only = velocity_only
        
        if data_dir:
            self.data_root = data_dir
        else:
            self.data_root = os.path.join(script_dir, "..", "data", self.symbol)
            
        logger.info(f"Target Data Directory: {os.path.abspath(self.data_root)}")

    def load_data_range(self, start_date: str, end_date: str):
        try:
            current = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD.")
            return None
        
        all_dfs = []
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = os.path.join(self.data_root, f"{date_str}.parquet")
            if os.path.exists(file_path):
                all_dfs.append(pd.read_parquet(file_path))
            current += timedelta(days=1)
            
        if not all_dfs: return None
        return pd.concat(all_dfs).sort_values("jump_id").reset_index(drop=True)

    def run_simulation(self, start_date: str, end_date: str, warmup_jumps: int = 100):
        df = self.load_data_range(start_date, end_date)
        if df is None:
            logger.error(f"No data found in {self.data_root} for the given range.")
            return

        total_rows = len(df)
        active_positions = []
        cumulative_pnl = 0.0
        wins, losses, max_overlap = 0, 0, 0
        
        # Tracking for duration calculation
        prev_server_ts = 0
        
        print("="*80)
        print(f" [ Demon Engine v2.3 - Coin Backtest Mode ({self.symbol}) ]")
        print(f" [ G:{self.target_gain} T:{self.base_threshold} W:{warmup_jumps} V:{self.engine.vol_multiplier} ]")
        print("="*80)

        for idx, row in df.iterrows():
            # 1. Exit Logic
            positions_to_close = [p for p in active_positions if row['jump_id'] >= p['target_jump_id']]
            active_positions = [p for p in active_positions if row['jump_id'] < p['target_jump_id']]
            
            for p in positions_to_close:
                direction = p['direction']
                exit_price = row['bid1'] if direction == 1 else row['ask1']
                net_pnl = direction * (exit_price - p['entry_price'])
                cumulative_pnl += net_pnl
                if net_pnl > 0: wins += 1
                else: losses += 1

            # 2. Historical Duration Calculation
            curr_ts = int(row['server_ts'])
            if prev_server_ts == 0:
                duration_ms = 100 # Initial gap fallback
            else:
                duration_ms = max(1, curr_ts - prev_server_ts)
            prev_server_ts = curr_ts

            # 3. State Reconstruction
            try:
                depth_data = json.loads(row['depth']) if isinstance(row['depth'], str) else row['depth']
                pot = MarketPotential(
                    asks=[PotentialEnergyLevel(p, q, True) for p, q in depth_data['asks']],
                    bids=[PotentialEnergyLevel(p, q, False) for p, q in depth_data['bids']]
                )
                
                raw_impacts = json.loads(row['impacts']) if isinstance(row['impacts'], str) else row['impacts']
                impacts = [KineticImpact(
                    offset_ms=imp.get('offset_ms', 1),
                    volume=imp.get('volume', 0),
                    is_buy=imp.get('is_buy', True),
                    p=imp.get('p', 0),
                    offset_ns=imp.get('offset_ns')
                ) for imp in raw_impacts]
                
                state = CoinJumpState(
                    jump_id=row['jump_id'], bid1=row['bid1'], ask1=row['ask1'],
                    server_ts=curr_ts, duration_ms=duration_ms, initial_potential=pot, 
                    impact_sequence=impacts, velocity_only=self.velocity_only
                )
            except Exception as e:
                logger.error(f"Data corruption at jump {row.get('jump_id', 'unknown')}: {e}")
                continue

            # 4. Engine Process
            # Important: Update matrix with current state first
            res = self.engine.process_state(state, horizon_n=1.0) 
            
            if idx < warmup_jumps: continue

            # Calculate Horizon N using the updated matrix
            matrix = res['matrix']
            horizon_n = calculate_physical_horizon(matrix, target_gain=self.target_gain)
            
            # Project probabilities for the derived horizon N
            m_future = self.engine.matrix_builder.get_n_step_matrix(horizon_n)
            
            dim = res['dimension']
            mid = dim // 2
            prob_dist = m_future[mid]
            
            if dim % 2 == 0:
                # Even (10x10): Perfectly split into 5:5 halves
                up_prob = np.sum(prob_dist[:mid])
                down_prob = np.sum(prob_dist[mid:])
            else:
                # Odd (5x5): Symmetric split skipping the center cell
                up_prob = np.sum(prob_dist[:mid])
                down_prob = np.sum(prob_dist[mid+1:])

            # [DEBUG PRINT] Monitor first 10 valid jumps after warmup
            if idx > warmup_jumps and idx < warmup_jumps + 10:
                print(f"Jump {idx} | qT: {res['quantized_T']} | qV: {res['quantized_V']} | Density: {res['density']:.8f} | N: {horizon_n:.1f}")
            
            # 5. Signal and Entry
            dynamic_threshold = self.base_threshold
            slippage_ticks = 1 if horizon_n > 200 else 3

            if up_prob > dynamic_threshold:
                entry_price = row['ask1'] + (slippage_ticks * self.tick_size)
                active_positions.append({
                    'target_jump_id': state.jump_id + int(horizon_n),
                    'direction': 1, 'entry_price': entry_price
                })
            elif down_prob > dynamic_threshold:
                entry_price = row['bid1'] - (slippage_ticks * self.tick_size)
                active_positions.append({
                    'target_jump_id': state.jump_id + int(horizon_n),
                    'direction': -1, 'entry_price': entry_price
                })
            
            max_overlap = max(max_overlap, len(active_positions))

        # [v2.3 Sync] Final Liquidation for accurate trade counting
        if active_positions:
            last_row = df.iloc[-1]
            for p in active_positions:
                direction = p['direction']
                exit_price = last_row['bid1'] if direction == 1 else last_row['ask1']
                price_diff = exit_price - p['entry_price']
                net_pnl = direction * price_diff
                cumulative_pnl += net_pnl
                if net_pnl > 0: wins += 1
                else: losses += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n" + "="*80)
        print(f" [ Final Report: {self.symbol} Quantum Simulation v2.3 ]")
        print("="*80)
        print(f"Total Evaluated Jumps : {len(df):,}")
        print(f"Total Trades Performed: {total_trades}")
        print(f"Wins: {wins:,} / Losses: {losses:,}")
        print(f"Win Rate              : {win_rate:.2f}%")
        print(f"Final Cumulative PnL  : {cumulative_pnl:+.2f} USDT")
        print(f"Max Concurrent Pos    : {max_overlap}")
        print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='btcusdt')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--t', type=float, default=0.83)
    parser.add_argument('--g', type=float, default=6.0)
    parser.add_argument('--v', action='store_true', help="Velocity Only Mode")
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--vol', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str)
    args = parser.parse_args()
    
    data_file = args.data_dir
    
    backtester = CoinBacktester(
        symbol=args.symbol, base_threshold=args.t, 
        target_gain=args.g, vol_multiplier=args.vol, velocity_only=args.v,
        data_dir=data_file
    )
    backtester.run_simulation(start_date=args.start, end_date=args.end, warmup_jumps=args.warmup)

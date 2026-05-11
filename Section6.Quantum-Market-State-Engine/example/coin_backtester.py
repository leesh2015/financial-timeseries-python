import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime, timedelta

# Production Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from coin_engine import CoinEngine, QuantumAdaptiveCore, TICK_SIZE
from coin_models import CoinJumpState, KineticImpact, PotentialEnergyLevel, MarketPotential, calculate_physical_horizon

logger = logging.getLogger(__name__)

class CoinBacktester:
    def __init__(self, symbol="btcusdt", base_threshold=0.83, target_gain=6.0, data_dir=None, vol_multiplier=1.0):
        self.symbol = symbol.lower()
        self.base_threshold = base_threshold
        self.target_gain = target_gain
        self.vol_multiplier = vol_multiplier
        self.tick_size = TICK_SIZE 
        
        self.engine = CoinEngine(vol_multiplier=vol_multiplier)
        self.adaptive_core = QuantumAdaptiveCore()
        
        self.data_root = data_dir or os.path.join(project_root, "data", self.symbol.upper())
        self.last_calib_jump = 0

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
            
            if os.path.isdir(file_path):
                import glob
                chunks = glob.glob(os.path.join(file_path, "*.parquet"))
                for c in chunks:
                    all_dfs.append(pd.read_parquet(c))
            elif os.path.exists(file_path):
                all_dfs.append(pd.read_parquet(file_path))
            
            current += timedelta(days=1)
            
        if not all_dfs: return None
        return pd.concat(all_dfs).sort_values("jump_id").reset_index(drop=True)

    def run_simulation(self, start_date: str, end_date: str, warmup_jumps: int = 100):
        df = self.load_data_range(start_date, end_date)
        if df is None:
            logger.error(f"No data found for the given range.")
            return

        total_rows = len(df)
        active_positions = []
        realized_pnl = 0.0
        wins, losses, max_overlap = 0, 0, 0
        total_longs, total_shorts = 0, 0
        horizon_history = [] 
        net_exposure_history = [] # Track (Longs - Shorts) over time
        
        prev_server_ts = 0
        tick_distribution = {}
        
        print("="*80)
        print(f" [ Quantum Backtester v3.4.8 (Zero-Constant) - {self.symbol.upper()} ]")
        print(f" [ Mode: High-Precision Physical Engine | G:{self.target_gain} T:{self.base_threshold} ]")
        print("="*80)


        for idx, row in df.iterrows():
            processed_jump_count = idx
            curr_mid = (row['bid1'] + row['ask1']) / 2.0
            
            self.adaptive_core.add_event(curr_mid)
            self.adaptive_core.current_tick = self.tick_size
            
            should_recalib = self.adaptive_core.check_coherence_collapse()
            if processed_jump_count - self.last_calib_jump > self.adaptive_core.current_window:
                should_recalib = True
            
            if should_recalib:
                old_tick = self.tick_size
                self.tick_size = self.adaptive_core.find_optimal_n(base_tick=TICK_SIZE)
                self.engine.hamiltonian.planck_hf = self.tick_size * 0.001 
                self.last_calib_jump = processed_jump_count
                
                print(f"[*] Calibration @ Jump {row['jump_id']} | W*: {self.adaptive_core.current_window} | "
                      f"λ: {self.adaptive_core.current_mfp:.1f} | Tc: {self.adaptive_core.current_tc:.1f} | "
                      f"Tick {old_tick:.2f} -> {self.tick_size:.2f}")

            tick_distribution[self.tick_size] = tick_distribution.get(self.tick_size, 0) + 1

            # 1. Position Exit
            positions_to_close = [p for p in active_positions if row['jump_id'] >= p['target_jump_id']]
            active_positions = [p for p in active_positions if row['jump_id'] < p['target_jump_id']]
            
            for p in positions_to_close:
                direction = p['direction']
                exit_price = row['bid1'] if direction == 1 else row['ask1']
                net_pnl = direction * (exit_price - p['entry_price'])
                realized_pnl += net_pnl
                if net_pnl > 0: wins += 1
                else: losses += 1

            # 2. State Construction & Engine Projection
            try:
                depth_data = json.loads(row['depth']) if isinstance(row['depth'], str) else row['depth']
                pot = MarketPotential(
                    asks=[PotentialEnergyLevel(p, q, True) for p, q in depth_data['asks']],
                    bids=[PotentialEnergyLevel(p, q, False) for p, q in depth_data['bids']]
                )
                raw_impacts = json.loads(row['impacts']) if isinstance(row['impacts'], str) else row['impacts']
                impacts = [KineticImpact(
                    offset_ms=imp.get('offset_ms', 1), volume=imp.get('volume', 0),
                    is_buy=imp.get('is_buy', True), p=imp.get('p', 0),
                    offset_ns=imp.get('offset_ns')
                ) for imp in raw_impacts]
                
                state = CoinJumpState(
                    jump_id=row['jump_id'], bid1=row['bid1'], ask1=row['ask1'],
                    server_ts=int(row['server_ts']), 
                    duration_ms=row.get('duration_ms', 100), # Use actual duration
                    initial_potential=pot, 
                    impact_sequence=impacts, spectral_gap=self.adaptive_core.spectral_gap
                )

                
                res = self.engine.process_state(state, target_gain=self.target_gain)
                self.adaptive_core.update_coherence_time(res['base_matrix'])
                
                if idx < warmup_jumps: continue

                horizon_n = res['horizon_n']
                horizon_history.append(horizon_n)
                m_future = res['matrix']
                
                prob_dist = m_future[res['dimension'] // 2]
                mid = res['dimension'] // 2
                if res['dimension'] % 2 == 0:
                    up_prob, down_prob = np.sum(prob_dist[:mid]), np.sum(prob_dist[mid:])
                else:
                    up_prob, down_prob = np.sum(prob_dist[:mid]), np.sum(prob_dist[mid+1:])
                
                # 3. Signal and Entry [v3.4 Zero-Constant Spec]
                # Spectral Gap based Dynamic Thresholding
                # If gap is small (chaos), dynamic_t increases to block noisy entries.
                spectral_gap = self.adaptive_core.spectral_gap
                q_v = res['quantized_V']
                
                # Dynamic Gamma derivation from Hamiltonian Spectral Gap
                dynamic_gamma = (1.0 - spectral_gap)
                dynamic_t = self.base_threshold + (dynamic_gamma / max(1.0, q_v))
                
                # Energy-based Dynamic Slippage
                # In BTC, slippage scales with energy density (T/V ratio)
                slippage_multiplier = np.clip(np.ceil(res['density']), 1, 5)
                slippage = (slippage_multiplier * 0.5) * self.tick_size
                
                if up_prob > dynamic_t:
                    total_longs += 1
                    active_positions.append({
                        'target_jump_id': state.jump_id + int(horizon_n),
                        'direction': 1, 'entry_price': row['ask1'] + slippage
                    })
                elif down_prob > dynamic_t:
                    total_shorts += 1
                    active_positions.append({
                        'target_jump_id': state.jump_id + int(horizon_n),
                        'direction': -1, 'entry_price': row['bid1'] - slippage
                    })
                
                max_overlap = max(max_overlap, len(active_positions))
                
                # 8. Record Net Exposure (L - S)
                current_l = sum(1 for p in active_positions if p['direction'] == 1)
                current_s = sum(1 for p in active_positions if p['direction'] == -1)
                net_exposure_history.append(current_l - current_s)
                
            except Exception:
                continue

        # Final Luxury Reporting
        total_realized = wins + losses
        realized_win_rate = (wins / total_realized * 100) if total_realized > 0 else 0.0
        
        floating_pnl = 0.0
        f_wins, f_losses = 0, 0
        last_row = df.iloc[-1]
        
        total_l_active = sum(1 for p in active_positions if p['direction'] == 1)
        total_s_active = sum(1 for p in active_positions if p['direction'] == -1)
        
        for p in active_positions:
            exit_p = last_row['bid1'] if p['direction'] == 1 else last_row['ask1']
            pnl = p['direction'] * (exit_p - p['entry_price'])
            floating_pnl += pnl
            if pnl > 0: f_wins += 1
            else: f_losses += 1

        c_total = total_realized + len(active_positions)
        c_wins, c_losses = wins + f_wins, losses + f_losses
        combined_win_rate = (c_wins / c_total * 100) if c_total > 0 else 0.0

        print(f"\nTotal Evaluated Jumps: {total_rows:,}")
        print(f"------------------------------------------------------------")
        print(f" [ 1. Realized Performance ]")
        print(f" Realized Trades : {total_realized} (L: {total_longs-total_l_active} / S: {total_shorts-total_s_active})")
        print(f" Realized WinRate: {realized_win_rate:.1f}% ({wins}W {losses}L)")
        print(f" Realized PnL    : {realized_pnl:+.2f} USDT")
        print(f"------------------------------------------------------------")
        print(f" [ 2. Floating Performance ]")
        print(f" Open Trades     : {len(active_positions)} (L: {total_l_active} / S: {total_s_active})")
        print(f" Floating WinRate: {(f_wins/len(active_positions)*100) if len(active_positions)>0 else 0:.1f}% ({f_wins}W {f_losses}L)")
        print(f" Floating PnL    : {floating_pnl:+.2f} USDT")
        print(f"------------------------------------------------------------")
        print(f" [ 3. Combined Total Performance ]")
        print(f" Total Trades    : {c_total}")
        print(f" Combined WinRate: {combined_win_rate:.1f}% ({c_wins}W {c_losses}L)")
        print(f" Combined Net PnL: {realized_pnl + floating_pnl:+.2f} USDT")
        print(f" Max Overlap     : {max_overlap}")
        if net_exposure_history:
            print(f" Net Exposure    : Max {np.max(net_exposure_history):+d} / Min {np.min(net_exposure_history):+d} / Avg {np.mean(net_exposure_history):+.2f}")
        if horizon_history:
            print(f" Horizon Stats   : Avg {np.mean(horizon_history):.1f} / Min {np.min(horizon_history):.1f} / Max {np.max(horizon_history):.1f}")
        print(f"------------------------------------------------------------")
        print(f" [ 4. Adaptive Tick Distribution ]")
        total_ticks = sum(tick_distribution.values())
        for t in sorted(tick_distribution.keys()):
            count = tick_distribution[t]
            print(f" Tick {t:.2f} : {count:5} steps ({(count/total_ticks*100):.1f}%)")
        avg_tick = sum(t * c for t, c in tick_distribution.items()) / total_ticks
        print(f" Average Tick: {avg_tick:.4f}")
        print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='btcusdt')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--t', type=float, default=0.83)
    parser.add_argument('--g', type=float, default=6.0)
    parser.add_argument('--vol-multiplier', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str)
    args = parser.parse_args()
    
    backtester = CoinBacktester(
        symbol=args.symbol, base_threshold=args.t, 
        target_gain=args.g, data_dir=args.data_dir,
        vol_multiplier=args.vol_multiplier
    )
    backtester.run_simulation(start_date=args.start, end_date=args.end)

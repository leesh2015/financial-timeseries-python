import sys
import os
import pandas as pd
import json
import numpy as np
import logging

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.quantum_engine import QuantumDemonEngine, QuantumAdaptiveCore, get_planck_unit
from core.quantum_models import KineticImpact, QuantumState, calculate_physical_horizon

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class QuantumBacktester:
    def __init__(self, data_path: str, tick_size: float = 0.01, threshold: float = 0.88, gain: float = 6.0, vol_multiplier: float = 2.0, max_units: int = 99999, velocity_only: bool = False, adaptive: bool = True):
        self.data_path = data_path
        self.vol_multiplier = vol_multiplier
        self.engine = QuantumDemonEngine(tick_size=tick_size, vol_multiplier=vol_multiplier)
        self.tick_size = tick_size 
        self.jump_scale = tick_size / 0.01
        self.threshold = threshold
        self.target_gain = gain
        self.max_units = max_units
        self.velocity_only = velocity_only 
        self.adaptive = adaptive
        self.adaptive_core = QuantumAdaptiveCore()
        self.last_calib_jump = 0

    def run_simulation(self, start_jump: int = 0):
        if not os.path.exists(self.data_path):
            logger.error("Data file not found.")
            return

        df = pd.read_parquet(self.data_path)
        total_rows = len(df)
        
        processed_jump_count = 0
        active_positions = [] 
        realized_pnl = 0.0
        wins, losses, max_overlap = 0, 0, 0
        total_entries = 0
        total_longs, total_shorts = 0, 0
        last_processed_price = None
        acc_impacts = []
        tick_distribution = {} # [v2.5] Track adaptive tick distribution
        horizon_history = [] # [v3.4.7] Track horizon N distribution


        mode_str = "Average Velocity (Old)" if self.velocity_only else "High-Precision Physical Engine (v3.0)"
        print("="*80)
        print(f" [ Quantum Backtester v3.4.8 (Zero-Constant) - Resolution: {self.tick_size} ]")
        print(f" [ Mode: {mode_str} | Scale: {self.jump_scale:.1f}x (Logic: Event-Time) ]")
        print("="*80)
        
        mids = ((df['bid1'] + df['ask1']) / 2.0).values
        for idx, (i, row) in enumerate(df.iterrows()):
            # 1. Closure Check (Based on Event-Time: processed_jump_count)
            remaining_positions = []
            for p in active_positions:
                if processed_jump_count >= p['target_jump_count']:
                    direction = p['direction']
                    exit_price = row['bid1'] if direction == 1 else row['ask1']
                    net_pnl = direction * (exit_price - p['entry_price'])
                    
                    realized_pnl += net_pnl
                    if net_pnl > 0: wins += 1
                    else: losses += 1
                else:
                    remaining_positions.append(p)
            active_positions = remaining_positions
            
            # 2. Energy Accumulation Logic
            raw_impacts = json.loads(row['impact_json']) if 'impact_json' in row else []
            acc_impacts.extend(raw_impacts)
            
            # Universal Resolution Trigger
            curr_mid = mids[idx]
            
            # [v3.0] Physical Coherence Collapse Loop
            if self.adaptive:
                self.adaptive_core.add_event(curr_mid)
                self.adaptive_core.current_tick = self.tick_size # Sync state
                # Watchdog interval = W* (physically derived from MFP, not a fixed constant)
                should_recalib = self.adaptive_core.check_coherence_collapse() or \
                                 (processed_jump_count - self.last_calib_jump > self.adaptive_core.current_window)
                
                if should_recalib:
                    new_tick = self.adaptive_core.find_optimal_n(base_tick=0.01)
                    if new_tick != self.tick_size:
                        mfp = self.adaptive_core.current_mfp
                        tc = self.adaptive_core.current_tc
                        w_star = self.adaptive_core.current_window
                        print(f"[*] Calibration @ Jump {processed_jump_count} | W*: {w_star:3d} | $\lambda$: {mfp:.1f} | Tc: {tc:5.1f} | Tick {self.tick_size:.2f} -> {new_tick:.2f}")
                        self.tick_size = new_tick
                        self.jump_scale = new_tick / 0.01
                        self.engine.tick_size = new_tick
                        self.engine.planck_hf = get_planck_unit(new_tick)
                        self.engine.hamiltonian.planck_hf = self.engine.planck_hf
                    self.last_calib_jump = processed_jump_count

            # Track tick distribution
            tick_distribution[self.tick_size] = tick_distribution.get(self.tick_size, 0) + 1

            if last_processed_price is not None:
                if abs(curr_mid - last_processed_price) < (self.tick_size - 1e-9):
                    continue

            # --- A Quantum Jump Occurs ---
            processed_jump_count += 1
            last_processed_price = curr_mid
            
            # 3. State Reconstruction
            impact_seq = []
            buy_v, sell_v = 0, 0
            for imp in acc_impacts:
                q_imp = KineticImpact(
                    offset_ms=imp.get("ms", 1), volume=imp.get("vol", 1),
                    is_buy=imp.get("buy", True), intensity=imp.get("intensity", 100.0),
                    offset_ns=imp.get("ns")
                )
                impact_seq.append(q_imp)
                if q_imp.is_buy: buy_v += q_imp.volume
                else: sell_v += q_imp.volume
            
            acc_impacts = [] # Clear for next jump
                
            # Use recorded duration for Velocity Only mode, force 0 for Acceleration mode
            if self.velocity_only:
                if 'duration_ms' in row:
                    d_ms = row['duration_ms']
                elif 'arrival_ns' in row and idx > 0:
                    prev_ns = df.iloc[idx-1].get('arrival_ns')
                    if prev_ns:
                        d_ms = max(1, (row['arrival_ns'] - prev_ns) // 1_000_000)
                    else:
                        d_ms = 1
                else:
                    d_ms = 1
            else:
                d_ms = 0
            
            state = QuantumState(
                jump_id=processed_jump_count, bid1=row['bid1'], ask1=row['ask1'],
                bid_vol1=row['bid_vol1'], ask_vol1=row['ask_vol1'],
                buy_vol=buy_v, sell_vol=sell_v, duration_ms=d_ms,
                impact_sequence=impact_seq
            )
            
            # 4. Engine Operation (v2.4 Unified Pipeline)
            res = self.engine.process_state(
                state, impact_seq, 
                target_gain=self.target_gain,
                velocity_only=self.velocity_only,
                jump_scale=self.jump_scale
            )
            
            q_v = res['quantized_V']
            horizon_n = res['horizon_n']
            horizon_history.append(horizon_n)
            powered_matrix = res['matrix']

            dim = res['dimension']
            mid = dim // 2
            prob_dist = powered_matrix[mid]
            
            # [v3.0] Update Coherence Time using the BASE transition matrix (not powered M^N)
            if self.adaptive:
                self.adaptive_core.update_coherence_time(self.engine.matrix_builder.matrix)
            
            # [v3.4.6 Hybrid] Signal Normalization for Entry (v3.0 style)
            base_matrix = self.engine.matrix_builder.matrix
            base_prob = base_matrix[mid]
            if dim % 2 == 0:
                raw_up, raw_dn = np.sum(base_prob[:mid]), np.sum(base_prob[mid:])
            else:
                raw_up, raw_dn = np.sum(base_prob[:mid]), np.sum(base_prob[mid+1:])
                
            total_p = raw_up + raw_dn
            if total_p > 1e-9:
                up_p, down_p = raw_up / total_p, raw_dn / total_p
            else:
                up_p, down_p = 0.5, 0.5


                
            # 5. Entry Logic (Zero-Constant Physical Friction Model)
            # Slippage is derived from Market Density (T/V ratio)
            # High Density = Thin book relative to impact = High Friction
            # [v3.4.5 Fix] Realistic Stock Slippage
            # Reduce from 1-5 to 0-1 for TQQQ liquidity
            slippage = int(np.clip(np.floor(res['density'] * 0.5), 0, 1))

            # [v3.1.9] Dynamic Gamma derivation from Hamiltonian Spectral Gap
            # Gamma = (1 - Spectral Gap) / max(1.0, q_v)
            # Physical Meaning: As the energy gap between states closes (Chaos), 
            # the barrier to entry increases proportionally.
            dynamic_gamma = (1.0 - res['spectral_gap'])
            dynamic_t = self.threshold + (dynamic_gamma / max(1.0, q_v))
            
            if idx >= start_jump:
                if up_p > dynamic_t and len(active_positions) < self.max_units:
                    entry_p = row['ask1'] + (slippage * 0.01)
                    active_positions.append({
                        'target_jump_count': processed_jump_count + int(horizon_n), 
                        'direction': 1, 'entry_price': entry_p
                    })
                    if total_longs % 10 == 0: print(f"[*] LONG Entry #{total_longs+1} at Jump {processed_jump_count} | Tick: {self.tick_size:.2f} | P: {up_p:.2f} > T: {dynamic_t:.2f}")
                    total_entries += 1
                    total_longs += 1
                elif down_p > dynamic_t and len(active_positions) < self.max_units:
                    entry_p = row['bid1'] - (slippage * 0.01)
                    active_positions.append({
                        'target_jump_count': processed_jump_count + int(horizon_n), 
                        'direction': -1, 'entry_price': entry_p
                    })
                    print(f"[*] SHORT Entry #{total_shorts+1} at Jump {processed_jump_count}")
                    total_entries += 1
                    total_shorts += 1
            
            max_overlap = max(max_overlap, len(active_positions))

        # 5. Final Report
        total_realized = wins + losses
        realized_win_rate = (wins / total_realized * 100) if total_realized > 0 else 0.0
        
        last_row = df.iloc[-1]
        floating_pnl = 0.0
        f_wins, f_losses = 0, 0
        for p in active_positions:
            direction = p['direction']
            exit_price = last_row['bid1'] if direction == 1 else last_row['ask1']
            pnl = direction * (exit_price - p['entry_price'])
            floating_pnl += pnl
            if pnl > 0: f_wins += 1
            else: f_losses += 1

        c_wins = wins + f_wins
        c_losses = losses + f_losses
        c_total = c_wins + c_losses
        combined_win_rate = (c_wins / c_total * 100) if c_total > 0 else 0.0

        print(f"Total Evaluated Rows: {total_rows:,}")
        print(f"------------------------------------------------------------")
        print(f" [ 1. Realized Performance ]")
        print(f" Realized Trades : {total_realized} (Long: {total_longs}, Short: {total_shorts})")
        print(f" Realized WinRate: {realized_win_rate:.1f}% ({wins}W {losses}L)")
        print(f" Realized PnL    : {realized_pnl:.2f} pts")
        print(f"------------------------------------------------------------")
        print(f" [ 2. Floating Performance ]")
        print(f" Open Trades     : {len(active_positions)}")
        print(f" Floating WinRate: {(f_wins/len(active_positions)*100) if len(active_positions)>0 else 0:.1f}% ({f_wins}W {f_losses}L)")
        print(f" Floating PnL    : {floating_pnl:.2f} pts")
        print(f"------------------------------------------------------------")
        print(f" [ 3. Combined Total Performance ]")
        print(f" Total Trades    : {c_total}")
        print(f" Combined WinRate: {combined_win_rate:.1f}% ({c_wins}W {c_losses}L)")
        print(f" Combined Net PnL: {realized_pnl + floating_pnl:.2f} pts")
        print(f" Max Overlap     : {max_overlap}")
        print(f"------------------------------------------------------------")
        
        if self.adaptive:
            print(f" [ 4. Adaptive Tick Distribution ]")
            total_ticks = sum(tick_distribution.values())
            for t in sorted(tick_distribution.keys()):
                count = tick_distribution[t]
                percentage = (count / total_ticks) * 100
                print(f" Tick {t:.2f} : {count:5} steps ({percentage:5.1f}%)")
            
            avg_tick = sum(t * c for t, c in tick_distribution.items()) / total_ticks
            print(f" Average Tick: {avg_tick:.4f}")
            
            if horizon_history:
                avg_h = sum(horizon_history) / len(horizon_history)
                min_h = min(horizon_history)
                max_h = max(horizon_history)
                print(f" Average Horizon N: {avg_h:.1f} (Min: {min_h}, Max: {max_h})")
            print(f"------------------------------------------------------------")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--tick', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.88)
    parser.add_argument('--gain', type=float, default=6.0)
    parser.add_argument('--units', type=int, default=99999)
    parser.add_argument('--vol-multiplier', type=float, default=2.0)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--velocity-only', action='store_true', help='Use legacy Average Velocity mode')
    parser.add_argument('--adaptive', action='store_true', default=True, help='Enable v3.0 Physical Lens (Adaptive)')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive', help='Disable physical adaptive logic')
    args = parser.parse_args()
    
    data_path = os.path.join(project_root, 'data', f'TQQQ_jump_data_{args.date}.parquet')
    backtester = QuantumBacktester(
        data_path, tick_size=args.tick, threshold=args.threshold, 
        gain=args.gain, vol_multiplier=args.vol_multiplier, 
        max_units=args.units, velocity_only=args.velocity_only,
        adaptive=args.adaptive
    )
    backtester.run_simulation(start_jump=args.warmup)

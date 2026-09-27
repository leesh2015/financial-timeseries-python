import sys
import os
import json
import logging
import asyncio
from collections import deque
from pathlib import Path
from redis.asyncio import Redis
import numpy as np
import time
from datetime import datetime

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.quantum_engine import QuantumDemonEngine, QuantumAdaptiveCore, get_planck_unit
from core.quantum_models import KineticImpact, QuantumState, calculate_physical_horizon

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class QuantumPredictor:
    def __init__(self, symbol="TQQQ", tick_size=0.01, base_threshold=0.83, target_gain=6.0, vol_multiplier=2.0, info_threshold=0.0, max_units=99999, warmup_jumps=0, velocity_only=False):
        self.symbol = symbol
        self.tick_size = tick_size
        self.jump_scale = tick_size / 0.01 if "BTC" not in symbol else tick_size / 0.1
        self.vol_multiplier = vol_multiplier
        self.velocity_only = velocity_only
        self.engine = QuantumDemonEngine(tick_size=tick_size, vol_multiplier=vol_multiplier)
        self.redis = None
        self.base_threshold = base_threshold
        self.target_gain = target_gain
        self.info_threshold = info_threshold
        self.max_units = max_units
        self.warmup_jumps = warmup_jumps
        self.ui_dirty = True # Mark UI as dirty for initial render
        
        # [v2.5] Adaptive Intelligence
        self.adaptive_core = QuantumAdaptiveCore()
        self.adaptive_enabled = True # Default to True for v2.5
        self.last_calib_jump = 0
        
        # Shared memory for UI
        self.shared_state = {
            'jump_id': 0,
            'bid1': 0.0,
            'ask1': 0.0,
            'density': 0.0,
            'up_prob': 0.0,
            'down_prob': 0.0,
            'threshold': 0.0,
            'signal': 'NEUTRAL',
            'tunneling_warning': False,
            'processed_events': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'entropy': 1.0,
            'adaptive_tick': tick_size,
            'calibrating': False,
            'unrealized_pnl': 0.0,
            'active_units': 0,
            'jumps_left': 0,
            'horizon_n': 0,
            'total_trades': 0,
            'long_count': 0,
            'short_count': 0,
            'win_rate': 0.0,
            'position': 'NONE'
        }
        
        self.active_positions = [] 
        self.last_entry_probs = None
        self.last_date = None
        self.logs = deque(maxlen=5)
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized.")
        
    def reset_session(self, new_date):
        """Reset state at the start of a new trading day"""
        msg = f"New session detected ({self.last_date} -> {new_date}). Resetting engine."
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.engine = QuantumDemonEngine(tick_size=self.tick_size, vol_multiplier=self.vol_multiplier)
        self.active_positions = []
        self.last_entry_probs = None
        self.last_date = new_date
        self.shared_state.update({
            'jump_id': 0, 'density': 0.0, 'up_prob': 0.0, 'down_prob': 0.0,
            'processed_events': 0, 'total_trades': 0, 'long_count': 0, 'short_count': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'wins': 0, 'losses': 0,
            'position': 'NONE', 'unrealized_pnl': 0.0, 'jumps_left': 0,
            'is_warmed_up': False
        })
        
    async def connect_redis(self):
        self.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await self.redis.ping()
        
    def _normalize_unitary(self):
        """Redundant in v2.4+ (Handled by Engine)"""
        pass

    async def engine_loop(self):
        await self.connect_redis()
        list_key = f"quantum:final_jumps:theory:{self.symbol}"
        logger.info(f"Quantum Predictor Running... waiting for {list_key}")
        
        while True:
            result = await self.redis.blpop(list_key, timeout=0)
            if not result: continue
            
            _, data_str = result
            try:
                jump_data = json.loads(data_str)
                jump_id = jump_data['jump_id']
                self.shared_state['last_update'] = datetime.now().strftime("%H:%M:%S")
                jump_date = jump_data.get('date')
                
                if (self.last_date and jump_date != self.last_date) or (jump_id < self.shared_state['jump_id'] - 50):
                    self.reset_session(jump_date)
                self.last_date = jump_date

                bid1, ask1 = jump_data['bid1'], jump_data['ask1']
                bid_vol1, ask_vol1 = jump_data['bid_vol1'], jump_data['ask_vol1']
                mid = (bid1 + ask1) / 2.0
                
                # 1. Adaptive Intelligence Layer (v2.5)
                self.adaptive_core.add_event(mid)
                
                # [v3.0] Physical Coherence Collapse Loop
                if self.adaptive_enabled:
                    self.adaptive_core.current_tick = self.tick_size # Sync state
                    # Watchdog interval = W* (physically derived from MFP, not a fixed constant)
                    should_recalib = self.adaptive_core.check_coherence_collapse() or \
                                     (jump_id - self.last_calib_jump > self.adaptive_core.current_window)
                    
                    if should_recalib and not self.adaptive_core.is_calibrating:
                        self.shared_state['calibrating'] = True
                        new_tick = self.adaptive_core.find_optimal_n(base_tick=0.01)
                        if new_tick != self.tick_size:
                            msg = f"Calibration: Tick {self.tick_size} -> {new_tick}"
                            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                            self.tick_size = new_tick
                            self.jump_scale = new_tick / 0.01 if "BTC" not in self.symbol else new_tick / 0.1
                            self.engine.tick_size = new_tick
                            self.engine.planck_hf = get_planck_unit(new_tick)
                            self.engine.hamiltonian.planck_hf = self.engine.planck_hf
                        
                        self.last_calib_jump = jump_id
                        self.shared_state['calibrating'] = False

                # 2. Restore impacts and energy
                impact_seq = []
                buy_v, sell_v = 0, 0
                for imp in jump_data['impacts']:
                    q_imp = KineticImpact(
                        offset_ms=imp.get("offset_ms") or imp.get("ms") or 1,
                        volume=imp.get("volume") or imp.get("vol") or 0.0,
                        is_buy=imp.get("is_buy") if "is_buy" in imp else imp.get("buy", True),
                        intensity=imp.get("intensity", 100.0),
                        offset_ns=imp.get("offset_ns") or imp.get("ns")
                    )
                    impact_seq.append(q_imp)
                    if q_imp.is_buy: buy_v += q_imp.volume
                    else: sell_v += q_imp.volume

                engine_state = QuantumState(
                    jump_id=jump_id, bid1=bid1, ask1=ask1,
                    bid_vol1=bid_vol1, ask_vol1=ask_vol1,
                    buy_vol=buy_v, sell_vol=sell_v, 
                    duration_ms=jump_data.get('duration_ms', 0),
                    impact_sequence=impact_seq
                )
                
                # 2. Engine computation (v2.4 Unified Pipeline)
                res = self.engine.process_state(
                    engine_state, impact_seq, 
                    velocity_only=self.velocity_only,
                    target_gain=self.target_gain,
                    jump_scale=self.jump_scale
                )
                
                q_v = res['quantized_V']
                density = res['density']
                horizon_n = res['horizon_n']
                powered_matrix = res['matrix']
                
                # [v3.0] Update Coherence Time using the BASE transition matrix (not powered M^N)
                if self.adaptive_enabled:
                    self.adaptive_core.update_coherence_time(self.engine.matrix_builder.matrix)
                
                # 3. Probability analysis
                dim = res['dimension']
                mid = dim // 2
                prob_dist = powered_matrix[mid]
                
                if dim % 2 == 0:
                    raw_up = np.sum(prob_dist[:mid])
                    raw_dn = np.sum(prob_dist[mid:])
                else:
                    raw_up = np.sum(prob_dist[:mid])
                    raw_dn = np.sum(prob_dist[mid+1:])
                
                # [v3.4.6 Hybrid] Signal Normalization for Entry (v3.0 style)
                # Normalizes UP/DOWN relative to each other ONLY for the signal,
                # ensuring the 0.83 threshold is reachable while Stay prob is high.
                total_prob = raw_up + raw_dn
                if total_prob > 1e-9:
                    up_prob, down_prob = raw_up / total_prob, raw_dn / total_prob
                else:
                    up_prob, down_prob = 0.5, 0.5


                
                # Sentiment Smoothing (v3.0) for visually stable dashboard
                alpha = 0.3
                prev_up = self.shared_state.get('up_prob', 0.5)
                prev_dn = self.shared_state.get('down_prob', 0.5)
                long_sentiment = (prev_up * (1 - alpha)) + (up_prob * alpha)
                short_sentiment = (prev_dn * (1 - alpha)) + (down_prob * alpha)
                
                # Ensure final sum is 1.0 for UI sentiment only
                s_total = long_sentiment + short_sentiment + 1e-9
                long_sentiment /= s_total
                short_sentiment /= s_total


                # [v3.4.7] Stable Physical Signal & Slippage (Backtest-Aligned)
                # Slippage is proportional to density but capped at 1 for TQQQ liquidity
                slippage = int(np.clip(np.floor(density * 0.5), 0, 1))
                
                # Use fixed threshold for precise entry as verified in backtesting
                # (Prevents noise from dynamic gamma/q_v fluctuations)
                final_threshold = self.base_threshold 

                
                # Information Innovation Filter (L1 Distance)
                is_new_info = True
                if self.info_threshold > 0:
                    if hasattr(self, 'last_entry_probs') and self.last_entry_probs is not None:
                        if prob_dist.shape == self.last_entry_probs.shape:
                            if np.sum(np.abs(prob_dist - self.last_entry_probs)) < self.info_threshold:
                                is_new_info = False
                
                # Warmup check
                is_warmed_up = True if self.warmup_jumps <= 0 else (jump_id >= self.warmup_jumps)
                
                # 4. Signal decision
                signal = 'NEUTRAL'
                if up_prob > final_threshold: signal = 'LONG'
                elif down_prob > final_threshold: signal = 'SHORT'

                
                # 5. Trading Simulation
                remaining_positions = []
                for pos in self.active_positions:
                    if jump_id >= pos['target_jump_id']:
                        exit_price = bid1 if pos['direction'] == 1 else ask1
                        pnl = pos['direction'] * (exit_price - pos['entry_price'])
                        self.shared_state['total_pnl'] += pnl
                        if pnl > 0: self.shared_state['wins'] += 1
                        else: self.shared_state['losses'] += 1
                    else:
                        remaining_positions.append(pos)
                self.active_positions = remaining_positions
                
                total_unrealized = 0.0
                for pos in self.active_positions:
                    curr_exit = bid1 if pos['direction'] == 1 else ask1
                    total_unrealized += pos['direction'] * (curr_exit - pos['entry_price'])

                if signal != 'NEUTRAL' and len(self.active_positions) < self.max_units and is_new_info and is_warmed_up:
                    direction = 1 if signal == 'LONG' else -1
                    entry_price = (ask1 + (slippage * 0.01)) if direction == 1 else (bid1 - (slippage * 0.01))
                    self.active_positions.append({
                        'direction': direction, 'entry_price': entry_price, 'target_jump_id': jump_id + horizon_n
                    })
                    self.last_entry_probs = prob_dist
                    self.shared_state['total_trades'] += 1
                    if direction == 1: self.shared_state['long_count'] += 1
                    else: self.shared_state['short_count'] += 1
                    self.shared_state['position'] = f"{signal} Entry"

                total_done = self.shared_state['wins'] + self.shared_state['losses']
                win_rate = (self.shared_state['wins'] / total_done * 100) if total_done > 0 else 0.0
                tunneling = (signal == 'LONG' and prob_dist[-1] > 0.3) or (signal == 'SHORT' and prob_dist[0] > 0.3)

                # 6. Atomic State Update
                self.shared_state.update({
                    'jump_id': int(jump_id), 'bid1': float(bid1), 'ask1': float(ask1), 'density': float(density),
                    'up_prob': float(long_sentiment), # Smoothed & Normalized
                    'down_prob': float(short_sentiment), # Smoothed & Normalized
                    'raw_up_prob': float(up_prob), # Raw (already normalized)
                    'raw_down_prob': float(down_prob),
                    'threshold': float(final_threshold),
                    'signal': signal, 'win_rate': float(win_rate), 'unrealized_pnl': float(total_unrealized),
                    'jumps_left': int(len(self.active_positions)), 'horizon_n': int(horizon_n),
                    'dimension': int(dim), 'tunneling': bool(tunneling), 'is_warmed_up': bool(is_warmed_up),
                    'mfp': float(self.adaptive_core.current_mfp), 'tc': float(self.adaptive_core.current_tc),
                    'tick': float(self.tick_size)
                })
                
                # 6. Update UI State (Internal Only)
                self.ui_dirty = True 
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")

    async def ui_loop(self):
        from rich.live import Live
        from rich.table import Table
        from rich.panel import Panel
        from rich.console import Console, Group
        from rich import box
        
        console = Console()
        def generate_dashboard():
            state = self.shared_state
            density = min(1.0, state['density'])
            density_color = "red" if density > 0.8 else "cyan"
            density_bar = f"[{density_color}]" + "█" * int(density * 20) + "░" * (20 - int(density * 20)) + "[/]"
            
            up_p, dn_p = min(1.0, state['up_prob']), min(1.0, state['down_prob'])
            up_bar = "[green]" + "█" * int(up_p * 15) + "░" * (15 - int(up_p * 15)) + f" {up_p*100:5.1f}%[/]"
            dn_bar = "[red]" + "█" * int(dn_p * 15) + "░" * (15 - int(dn_p * 15)) + f" {dn_p*100:5.1f}%[/]"
            
            sig = state['signal']
            sig_color = "bold green" if sig == 'LONG' else ("bold red" if sig == 'SHORT' else "white")
            
            table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
            table.add_row("Engine Status", f"[bold green]RUNNING[/] (Last Data: [yellow]{state.get('last_update', 'WAITING')}[/])")
            table.add_row("Jump ID", f"[bold white]{state['jump_id']}[/]")
            table.add_row("Current Price", f"Ask: [bold red]{state['ask1']:.2f}[/] / Bid: [bold blue]{state['bid1']:.2f}[/]")
            table.add_row("Energy Density", f"{density_bar} {state['density']:.4f}")
            table.add_row("Quantum Dimension", f"[bold magenta]{state.get('dimension', 5)}x{state.get('dimension', 5)}[/] (Hilbert Space)")
            table.add_row("Horizon N", f"[bold yellow]{state['horizon_n']}[/] Jumps (Spacetime Curvature)")
            table.add_row("UP Prob", up_bar)
            table.add_row("DOWN Prob", dn_bar)
            
            sig_text = f"[{sig_color}]{sig}[/]"
            if state['tunneling_warning']: sig_text += " [blink bold yellow]⚠️ TUNNELING[/]"
            table.add_row("ENGINE SIGNAL", sig_text)

            pnl = state['unrealized_pnl']
            pnl_color = "bold bright_green" if pnl >= 0 else "bold bright_red"
            pnl_panel = Panel(
                f"[{pnl_color}]{pnl:+.2f} pts[/]",
                title="[bold white]FLOATING PnL[/]",
                border_style=pnl_color,
                padding=(1, 4),
                subtitle=f"[dim]Total PnL: {state['total_pnl']:+.2f}[/]"
            )

            perf_table = Table(box=box.SIMPLE, show_header=True, expand=True, border_style="dim")
            perf_table.add_column("Metrics", style="cyan")
            perf_table.add_column("Values", justify="right")
            perf_table.add_row("Open Trades", f"[bold magenta]{state['jumps_left']}[/] (Superposition)")
            perf_table.add_row("Total Trades", f"{state['total_trades']} (L:{state['long_count']} S:{state['short_count']})")
            perf_table.add_row("Win Rate", f"[bold yellow]{state['win_rate']:.1f}%[/] ({state['wins']}W {state['losses']}L)")
            perf_table.add_row("Status", f"[bold white]{state['position']}[/]")

            top_group = Group(
                Panel(table, title="[bold blue]Market Quantum State[/]", border_style="blue"),
                pnl_panel
            )
            
            log_panel = Panel(
                "\n".join(self.logs),
                title="[bold white]System Logs[/]",
                border_style="dim",
                padding=(0, 1)
            )

            return Panel(
                Group(
                    top_group, 
                    Panel(perf_table, title="[bold yellow]Performance Dashboard[/]", border_style="yellow"),
                    log_panel
                ),
                title=f"[bold cyan]Quantum Engine v3.4.8 (Zero-Constant) (G:{self.target_gain} T:{self.base_threshold} Tick:{self.tick_size:.2f})[/]",
                subtitle=f"[dim]Units: {len(self.active_positions)}/{self.max_units} | Clock: {datetime.now().strftime('%H:%M:%S')} | Live[/]",
                border_style="bright_cyan",
                padding=(1, 2)
            )

        # Use auto_refresh=True and higher refresh rate to ensure the terminal stays alive
        # Use screen=True to eliminate flickering and afterimages completely
        with Live(generate_dashboard(), console=console, auto_refresh=True, refresh_per_second=4, screen=True) as live:
            while True:
                await asyncio.sleep(0.2) # Regular heartbeat check
                # Even with auto_refresh, we can still manually trigger update on new data
                if self.ui_dirty:
                    live.update(generate_dashboard())
                    self.ui_dirty = False

    async def run(self):
        await asyncio.gather(self.engine_loop(), self.ui_loop())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, help="Symbol (e.g. TQQQ)")
    parser.add_argument('--tick', type=float, help="Tick Size for Aggregation (e.g. 0.05)")
    parser.add_argument('--threshold', type=float, help="Base entry threshold")
    parser.add_argument('--gain', type=float, help="Target gain for horizon")
    parser.add_argument('--info', type=float, help="Information Innovation threshold")
    parser.add_argument('--units', type=int, help="Max concurrent units")
    parser.add_argument('--vol-multiplier', type=float, help="Volume Multiplier (Sampling Compensation)")
    parser.add_argument('--warmup', type=int, help="Warmup jumps")
    parser.add_argument('--velocity-only', action='store_true', help="Force constant velocity mode")
    args = parser.parse_args()
    
    # Filter out None values to let Class defaults handle them
    params = {k: v for k, v in vars(args).items() if v is not None}
    # Map argparse keys to constructor keys if necessary
    if 'threshold' in params: params['base_threshold'] = params.pop('threshold')
    if 'tick' in params: params['tick_size'] = params.pop('tick')
    if 'gain' in params: params['target_gain'] = params.pop('gain')
    if 'info' in params: params['info_threshold'] = params.pop('info')
    if 'warmup' in params: params['warmup_jumps'] = params.pop('warmup')
    if 'units' in params: params['max_units'] = params.pop('units')
    
    predictor = QuantumPredictor(**params)
    try: asyncio.run(predictor.run())
    except KeyboardInterrupt: print("\nTerminated.")

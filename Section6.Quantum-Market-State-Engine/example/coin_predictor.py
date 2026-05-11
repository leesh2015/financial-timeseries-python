import asyncio
import json
import time
import argparse
import os
import sys
import logging
import warnings
from redis.asyncio import Redis
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

# Production Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from coin_engine import CoinEngine, QuantumAdaptiveCore, TICK_SIZE
from coin_models import CoinJumpState, KineticImpact, PotentialEnergyLevel, MarketPotential, calculate_physical_horizon

class CoinPredictor:
    def __init__(self, symbol="btcusdt", threshold=0.83, gain=6.0, vol=1.0, units=99999):
        self.symbol = symbol.lower()
        self.base_threshold = threshold
        self.target_gain = gain
        self.vol_multiplier = vol
        self.max_units = units
        self.tick_size = TICK_SIZE
        
        self.engine = CoinEngine(vol_multiplier=vol)
        self.adaptive_core = QuantumAdaptiveCore()
        self.redis = None
        self.console = Console()
        self.last_calib_jump = 0
        self.ui_dirty = True
        
        self.shared_state = {
            'jump_id': 0, 'bid1': 0.0, 'ask1': 0.0, 'density': 0.0,
            'up_prob': 0.5, 'down_prob': 0.5, 'threshold': threshold,
            'signal': 'NEUTRAL', 'win_rate': 0.0, 'total_pnl': 0.0, 
            'wins': 0, 'losses': 0, 'position': 'WAITING', 
            'unrealized_pnl': 0.0, 'jumps_left': 0, 'horizon_n': 0, 
            'dimension': 5, 'mfp': 1.0, 'tc': 100.0, 'tick': TICK_SIZE
        }
        self.active_positions = []

    async def connect_redis(self):
        self.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await self.redis.ping()

    def generate_dashboard(self):
        state = self.shared_state
        density = min(1.0, state['density'])
        density_bar = f"[cyan]" + "█" * int(density * 20) + "░" * (20 - int(density * 20)) + "[/]"
        
        up_p, dn_p = min(1.0, state['up_prob']), min(1.0, state['down_prob'])
        up_bar = "[green]" + "█" * int(up_p * 15) + "░" * (15 - int(up_p * 15)) + f" {up_p*100:5.1f}%[/]"
        dn_bar = "[red]" + "█" * int(dn_p * 15) + "░" * (15 - int(dn_p * 15)) + f" {dn_p*100:5.1f}%[/]"
        
        sig = state['signal']
        sig_color = "bold green" if sig == 'LONG' else ("bold red" if sig == 'SHORT' else "white")
        
        table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
        table.add_row("Jump ID", f"[bold white]{state['jump_id']}[/]")
        table.add_row("Price", f"Ask: [bold red]{state['ask1']:.2f}[/] / Bid: [bold blue]{state['bid1']:.2f}[/]")
        table.add_row("Energy Density", f"{density_bar} {state['density']:.4f}")
        table.add_row("Mean Free Path", f"[bold yellow]{state['mfp']:.2f}[/] (Collision Rate)")
        table.add_row("Coherence Time", f"[bold magenta]{state['tc']:.1f}[/] (Wave Life)")
        table.add_row("Horizon N", f"[bold white]{state['horizon_n']}[/]")
        table.add_row("UP Prob", up_bar)
        table.add_row("DOWN Prob", dn_bar)
        table.add_row("ENGINE SIGNAL", f"[{sig_color}]{sig}[/]")

        pnl = state['unrealized_pnl']
        pnl_color = "bold bright_green" if pnl >= 0 else "bold bright_red"
        pnl_panel = Panel(
            f"[{pnl_color}]{pnl:+.2f} USDT[/]",
            title="[bold white]FLOATING PnL[/]",
            border_style=pnl_color,
            padding=(1, 4),
            subtitle=f"[dim]Total PnL: {state['total_pnl']:+.2f}[/]"
        )

        perf_table = Table(box=box.SIMPLE, show_header=True, expand=True, border_style="dim")
        perf_table.add_column("Metrics", style="cyan")
        perf_table.add_column("Values", justify="right")
        perf_table.add_row("Open Units", f"[bold magenta]{state['jumps_left']}[/]")
        perf_table.add_row("Win Rate", f"[bold yellow]{state['win_rate']:.1f}%[/] ({state['wins']}W {state['losses']}L)")
        perf_table.add_row("Status", f"[bold white]{state['position']}[/]")

        top_group = Group(
            Panel(table, title="[bold blue]Market Quantum State[/]", border_style="blue"),
            pnl_panel
        )
        
        return Panel(
            Group(top_group, Panel(perf_table, title="[bold yellow]Performance Dashboard[/]", border_style="yellow")),
            title=f"[bold cyan]Quantum Coin Predictor v3.4.8 (Zero-Constant) (Tick:{state['tick']:.2f})[/]",
            subtitle=f"[dim]Symbol: {self.symbol.upper()} | Physical Lens | Live[/]",

            border_style="bright_cyan",
            padding=(1, 2)
        )

    async def engine_loop(self):
        await self.connect_redis()
        list_key = f"quantum:final_jumps:theory:{self.symbol}"
        
        while True:
            try:
                result = await self.redis.blpop(list_key, timeout=0)
                if not result: continue
                
                jump_data = json.loads(result[1])
                curr_mid = (jump_data['bid1'] + jump_data['ask1']) / 2.0
                
                # 1. Physical Core Update
                self.adaptive_core.add_event(curr_mid)
                self.adaptive_core.current_tick = self.tick_size
                
                # 2. Dynamic Calibration
                should_recalib = self.adaptive_core.check_coherence_collapse()
                if jump_data['jump_id'] - self.last_calib_jump > self.adaptive_core.current_window:
                    should_recalib = True
                
                if should_recalib:
                    self.tick_size = self.adaptive_core.find_optimal_n(base_tick=TICK_SIZE)
                    self.engine.hamiltonian.planck_hf = self.tick_size * 0.001
                    self.last_calib_jump = jump_data['jump_id']
                
                # 3. State Processing
                pot = MarketPotential(
                    asks=[PotentialEnergyLevel(p, q, True) for p, q in jump_data['depth']['asks']],
                    bids=[PotentialEnergyLevel(p, q, False) for p, q in jump_data['depth']['bids']]
                )
                impacts = [KineticImpact(
                    offset_ms=imp.get("offset_ms") or imp.get("ms") or 1,
                    volume=imp.get("volume") or imp.get("vol") or 0.0,
                    is_buy=imp.get("is_buy") if "is_buy" in imp else imp.get("buy", True),
                    p=imp.get('p', 0),
                    offset_ns=imp.get('offset_ns') or imp.get('ns')
                ) for imp in jump_data['impacts']]

                state = CoinJumpState(
                    jump_id=jump_data['jump_id'], bid1=jump_data['bid1'], ask1=jump_data['ask1'],
                    server_ts=jump_data['server_ts'], 
                    duration_ms=jump_data.get('duration_ms', 100), 
                    initial_potential=pot, impact_sequence=impacts,
                    spectral_gap=self.adaptive_core.spectral_gap
                )

                
                res = self.engine.process_state(state, target_gain=self.target_gain)
                self.adaptive_core.update_coherence_time(res['base_matrix'])
                
                horizon_n = res['horizon_n']
                m_future = res['matrix']
                
                dim = res['dimension']
                mid = dim // 2
                prob_dist = m_future[mid]
                if dim % 2 == 0:
                    raw_up, raw_dn = np.sum(prob_dist[:mid]), np.sum(prob_dist[mid:])
                else:
                    raw_up, raw_dn = np.sum(prob_dist[:mid]), np.sum(prob_dist[mid+1:])
                
                # [v3.4 Fix] Unified Probability Spec (No Normalization)
                # Matches backtester's precision by accounting for 'Stay' probability.
                up_prob, down_prob = raw_up, raw_dn

                
                # 3.1. [v3.4 Spec] Dynamic Thresholding & Signal Generation
                # Use spectral gap and energy density to filter noise
                spectral_gap = self.adaptive_core.spectral_gap
                q_v = res['quantized_V']
                dynamic_gamma = (1.0 - spectral_gap)
                dynamic_t = self.base_threshold + (dynamic_gamma / max(1.0, q_v))
                
                signal = 'NEUTRAL'
                if up_prob > dynamic_t: signal = 'LONG'
                elif down_prob > dynamic_t: signal = 'SHORT'
                
                # 4. Position Simulation
                current_unrealized = 0.0
                remaining_positions = []
                for pos in self.active_positions:
                    if state.jump_id >= pos['target_jump_id']:
                        exit_price = state.bid1 if pos['direction'] == 1 else state.ask1
                        pnl = pos['direction'] * (exit_price - pos['entry_price'])
                        self.shared_state['total_pnl'] += pnl
                        if pnl > 0: self.shared_state['wins'] += 1
                        else: self.shared_state['losses'] += 1
                    else:
                        curr_exit = state.bid1 if pos['direction'] == 1 else state.ask1
                        current_unrealized += pos['direction'] * (curr_exit - pos['entry_price'])
                        remaining_positions.append(pos)
                self.active_positions = remaining_positions
                
                # Energy-based Dynamic Slippage (Parity with Backtester)
                slippage_multiplier = np.clip(np.ceil(res['density']), 1, 5)
                slippage = (slippage_multiplier * 0.5) * self.tick_size
                
                if signal != 'NEUTRAL' and len(self.active_positions) < self.max_units:
                    direction = 1 if signal == 'LONG' else -1
                    entry_price = (state.ask1 + slippage) if direction == 1 else (state.bid1 - slippage)
                    self.active_positions.append({
                        'direction': direction, 'entry_price': entry_price, 
                        'target_jump_id': state.jump_id + int(horizon_n)
                    })

                # [v3.0] Frontend Sentiment: Relative Position Ratio (0.0 ~ 1.0)
                current_l = sum(1 for p in self.active_positions if p['direction'] == 1)
                current_s = sum(1 for p in self.active_positions if p['direction'] == -1)
                total_active = current_l + current_s
                
                if total_active > 0:
                    long_sentiment = current_l / total_active
                    short_sentiment = current_s / total_active
                total_done = self.shared_state['wins'] + self.shared_state['losses']
                win_rate = (self.shared_state['wins'] / total_done * 100) if total_done > 0 else 0.0

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


                # 5. Atomic State Update
                self.shared_state.update({
                    'jump_id': int(state.jump_id), 'bid1': float(state.bid1), 'ask1': float(state.ask1),
                    'up_prob': float(long_sentiment), 'down_prob': float(short_sentiment),
                    'raw_up_prob': float(up_prob), 'raw_down_prob': float(down_prob),
                    'threshold': float(dynamic_t), 'signal': signal,

                    'win_rate': float(win_rate), 'unrealized_pnl': float(current_unrealized),
                    'jumps_left': int(len(self.active_positions)), 'horizon_n': int(horizon_n),
                    'dimension': int(dim), 'tick': float(self.tick_size),
                    'density': res['density'], 'mfp': self.adaptive_core.current_mfp, 
                    'tc': self.adaptive_core.current_tc,
                    'position': 'TRADING' if len(self.active_positions) > 0 else 'WAITING'
                })
                
                # 5. Update UI State (Internal Only)
                self.ui_dirty = True

            except Exception:
                await asyncio.sleep(1)

    async def ui_loop(self):
        with Live(self.generate_dashboard(), console=self.console, refresh_per_second=4, screen=True) as live:
            while True:
                await asyncio.sleep(0.2)
                if self.ui_dirty:
                    live.update(self.generate_dashboard())
                    self.ui_dirty = False

    async def run(self):
        await asyncio.gather(self.engine_loop(), self.ui_loop())

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="Quantum Coin Predictor v3.4.8 (Zero-Constant)")
    parser.add_argument('--symbol', type=str, default='btcusdt')
    parser.add_argument('--g', type=float, default=6.0)
    parser.add_argument('--t', type=float, default=0.83)
    args = parser.parse_args()

    predictor = CoinPredictor(symbol=args.symbol, gain=args.g, threshold=args.t)
    try: asyncio.run(predictor.run())
    except KeyboardInterrupt: pass

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

# Add parent directory to path to use core logic if needed, 
# but here we use the local coin_engine/models for standalone example purity.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from coin_engine import CoinEngine
from coin_models import CoinJumpState, KineticImpact, PotentialEnergyLevel, MarketPotential, calculate_physical_horizon

class CoinPredictor:
    def __init__(self, symbol="btcusdt", threshold=0.83, gain=6.0, info=0.03, vol=1.0, units=99999, velocity_only=False):
        self.symbol = symbol.lower()
        self.base_threshold = threshold
        self.target_gain = gain
        self.info_threshold = info
        self.vol_multiplier = vol
        self.max_units = units
        self.velocity_only = velocity_only
        
        self.engine = CoinEngine(vol_multiplier=vol)
        self.redis = None
        self.console = Console()
        
        self.shared_state = {
            'jump_id': 0, 'bid1': 0.0, 'ask1': 0.0, 'density': 0.0,
            'up_prob': 0.5, 'down_prob': 0.5, 'threshold': threshold,
            'signal': 'NEUTRAL', 'tunneling_warning': False,
            'total_trades': 0, 'long_count': 0, 'short_count': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'wins': 0, 'losses': 0,
            'position': 'WAITING', 'unrealized_pnl': 0.0, 'active_units': 0,
            'horizon_n': 0, 'dimension': 5, 'latency_ms': 0
        }
        self.active_positions = []
        self.last_up_prob = 0.5

    async def connect_redis(self):
        self.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await self.redis.ping()

    def generate_dashboard(self):
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
        table.add_row("Jump ID", f"[bold white]{state['jump_id']}[/]")
        table.add_row("Price", f"Ask: [bold red]{state['ask1']:.2f}[/] / Bid: [bold blue]{state['bid1']:.2f}[/]")
        table.add_row("Energy Density", f"{density_bar} {state['density']:.4f}")
        table.add_row("Hilbert Space", f"[bold magenta]{state['dimension']}x{state['dimension']}[/]")
        table.add_row("Horizon N", f"[bold yellow]{state['horizon_n']:.2f}[/] (Curvature)")
        table.add_row("UP Prob", up_bar)
        table.add_row("DOWN Prob", dn_bar)
        
        sig_text = f"[{sig_color}]{sig}[/]"
        if state['tunneling_warning']: sig_text += " [blink bold yellow]⚠️ TUNNELING[/]"
        table.add_row("ENGINE SIGNAL", sig_text)

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
        perf_table.add_row("Open Units", f"[bold magenta]{state['active_units']}[/] (Superposition)")
        perf_table.add_row("Total Trades", f"{state['total_trades']} (L:{state['long_count']} S:{state['short_count']})")
        perf_table.add_row("Win Rate", f"[bold yellow]{state['win_rate']:.1f}%[/] ({state['wins']}W {state['losses']}L)")
        perf_table.add_row("Status", f"[bold white]{state['position']}[/] ({state['latency_ms']}ms)")

        top_group = Group(
            Panel(table, title="[bold blue]Market Quantum State[/]", border_style="blue"),
            pnl_panel
        )
        
        return Panel(
            Group(top_group, Panel(perf_table, title="[bold yellow]Performance Dashboard[/]", border_style="yellow")),
            title=f"[bold cyan]Demon Engine v2.3 (EVENT-DRIVEN) (G:{self.target_gain} T:{self.base_threshold} U:{self.max_units})[/]",
            subtitle=f"[dim]Symbol: {self.symbol.upper()} | High-Precision Mode | Theory[/]",
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
                
                pot = MarketPotential(
                    asks=[PotentialEnergyLevel(p, q, True) for p, q in jump_data['depth']['asks']],
                    bids=[PotentialEnergyLevel(p, q, False) for p, q in jump_data['depth']['bids']]
                )
                
                impacts = []
                for imp in jump_data['impacts']:
                    impacts.append(KineticImpact(
                        offset_ms=imp.get('ms', 1),
                        volume=imp.get('vol', 0),
                        is_buy=imp.get('buy', True),
                        p=imp.get('p', 0),
                        offset_ns=imp.get('ns') # v2.3 high-precision
                    ))

                state = CoinJumpState(
                    jump_id=jump_data['jump_id'], bid1=jump_data['bid1'], ask1=jump_data['ask1'],
                    server_ts=jump_data['server_ts'], initial_potential=pot, 
                    impact_sequence=impacts, velocity_only=self.velocity_only
                )
                
                # Use current matrix to derive horizon N (High-precision v2.3)
                # Derive horizon N (Zero Lag Architecture)
                # First, process with N=1 to get the most current matrix state
                res = self.engine.process_state(state, horizon_n=1.0)
                current_matrix = res['matrix']
                horizon_n = calculate_physical_horizon(current_matrix, target_gain=self.target_gain)
                
                # Project probabilities for the newly derived horizon N
                powered_matrix = self.engine.matrix_builder.get_n_step_matrix(horizon_n)
                dim = res['dimension']
                mid = dim // 2
                prob_dist = powered_matrix[mid]
                
                if dim % 2 == 0:
                    # Even (10x10): Perfectly split into 5:5 halves
                    up_prob = np.sum(prob_dist[:mid])
                    down_prob = np.sum(prob_dist[mid:])
                else:
                    # Odd (5x5): Symmetric split skipping the center cell
                    up_prob = np.sum(prob_dist[:mid])
                    down_prob = np.sum(prob_dist[mid+1:])
                
                # Information Innovation Filter
                prob_diff = abs(up_prob - self.last_up_prob)
                self.last_up_prob = up_prob
                
                signal = 'NEUTRAL'
                if up_prob > self.base_threshold and prob_diff > self.info_threshold: signal = 'LONG'
                elif down_prob > self.base_threshold and prob_diff > self.info_threshold: signal = 'SHORT'
                
                # Trading Simulation
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
                
                if signal != 'NEUTRAL' and len(self.active_positions) < self.max_units:
                    direction = 1 if signal == 'LONG' else -1
                    entry_price = state.ask1 if direction == 1 else state.bid1
                    self.active_positions.append({
                        'direction': direction, 'entry_price': entry_price, 
                        'target_jump_id': state.jump_id + horizon_n
                    })
                    self.shared_state['total_trades'] += 1
                    if direction == 1: self.shared_state['long_count'] += 1
                    else: self.shared_state['short_count'] += 1

                total_done = self.shared_state['wins'] + self.shared_state['losses']
                win_rate = (self.shared_state['wins'] / total_done * 100) if total_done > 0 else 0.0
                tunneling = (up_prob > 0.4 and prob_dist[0] > 0.1) or (down_prob > 0.4 and prob_dist[-1] > 0.1)
                
                # Latency from server arrival
                latency = (time.time_ns() // 1_000_000) - state.server_ts

                self.shared_state.update({
                    'jump_id': state.jump_id, 'bid1': state.bid1, 'ask1': state.ask1,
                    'density': res['density'], 'up_prob': up_prob, 'down_prob': down_prob,
                    'signal': signal, 'win_rate': win_rate, 'unrealized_pnl': current_unrealized,
                    'active_units': len(self.active_positions), 'horizon_n': horizon_n,
                    'dimension': dim, 'tunneling_warning': tunneling, 'latency_ms': latency,
                    'position': 'TRADING' if len(self.active_positions) > 0 else 'WAITING'
                })

            except Exception as e:
                logging.error(f"Engine Loop Error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def ui_loop(self):
        with Live(self.generate_dashboard(), console=self.console, refresh_per_second=4, screen=True) as live:
            while True:
                await asyncio.sleep(0.25)
                try: live.update(self.generate_dashboard())
                except Exception: pass

    async def run(self):
        await asyncio.gather(self.engine_loop(), self.ui_loop())

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    # Enable Windows VT100 mode
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception: pass

    logging.basicConfig(filename='engine.log', level=logging.ERROR,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='btcusdt')
    parser.add_argument('--g', type=float, default=6.0)
    parser.add_argument('--t', type=float, default=0.83)
    parser.add_argument('--i', type=float, default=0.03)
    parser.add_argument('--v', action='store_true', help="Velocity Only Mode")
    parser.add_argument('--u', type=int, default=99999)
    args = parser.parse_args()

    predictor = CoinPredictor(
        symbol=args.symbol, gain=args.g, threshold=args.t, 
        info=args.i, units=args.u, velocity_only=args.v
    )

    try:
        asyncio.run(predictor.run())
    except KeyboardInterrupt:
        pass

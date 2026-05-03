import sys
import os
import json
import logging
import asyncio
import time
from pathlib import Path
import numpy as np

# Add project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.quantum_engine import QuantumDemonEngine, QuantumState
from core.quantum_models import KineticImpact, calculate_physical_horizon

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class QuantumPredictor:
    def __init__(self, symbol="QQQ", host='127.0.0.1', port=9999, 
                 base_threshold=0.83, target_gain=10.15, info_threshold=0.03, 
                 vol_multiplier=1.0, max_units=100, warmup_jumps=100, velocity_only=False):
        self.symbol = symbol
        self.host = host
        self.port = port
        self.base_threshold = base_threshold
        self.target_gain = target_gain
        self.info_threshold = info_threshold
        self.vol_multiplier = vol_multiplier
        self.max_units = max_units
        self.warmup_jumps = warmup_jumps
        self.velocity_only = velocity_only
        
        self.engine = QuantumDemonEngine(vol_multiplier=self.vol_multiplier)
        self.last_date = None
        self.last_jump_id = -1
        self.last_up_prob = 0.5
        
        # Shared Memory (State for UI)
        self.shared_state = {
            'jump_id': 0,
            'bid1': 0.0,
            'ask1': 0.0,
            'density': 0.0,
            'up_prob': 0.5,
            'down_prob': 0.5,
            'threshold': 0.0,
            'signal': 'NEUTRAL',
            'tunneling_warning': False,
            'processed_events': 0,
            'total_trades': 0,
            'long_count': 0,
            'short_count': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'position': 'WARMING UP',
            'unrealized_pnl': 0.0,
            'active_units': 0,
            'horizon_n': 0,
            'dimension': 5
        }
        
        self.active_positions = [] 
        self.tick_size = 0.01
        
    def reset_session(self, new_date):
        """Reset the engine and statistics for a new trading session"""
        logger.warning(f"New session detected for date: {new_date}. Resetting state...")
        self.engine = QuantumDemonEngine(vol_multiplier=self.vol_multiplier)
        self.active_positions = []
        self.last_date = new_date
        self.last_up_prob = 0.5
        
        # Reset UI statistics (keep total performance)
        self.shared_state.update({
            'jump_id': 0,
            'density': 0.0,
            'up_prob': 0.5,
            'down_prob': 0.5,
            'signal': 'NEUTRAL',
            'tunneling_warning': False,
            'position': 'WARMING UP',
            'unrealized_pnl': 0.0,
            'active_units': 0,
            'horizon_n': 0
        })
        
    async def engine_loop(self):
        logger.warning(f"Connecting to Mock Collector at {self.host}:{self.port}...")
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
                logger.warning(f"Connected to Live Stream for {self.symbol}!")
                break
            except ConnectionRefusedError:
                logger.info("Connection refused. Waiting for mock_collector.py to start...")
                await asyncio.sleep(2)
        
        while True:
            try:
                line = await reader.readline()
                if not line:
                    logger.warning("Stream ended.")
                    break
                    
                data_str = line.decode('utf-8').strip()
                if not data_str: continue
                
                jump_data = json.loads(data_str)
                if jump_data.get("EOF"):
                    logger.warning("End of file reached in Mock Collector.")
                    break
                
                # Check for session change
                curr_date = jump_data.get('date')
                jump_id = jump_data['jump_id']
                
                if (self.last_date and curr_date != self.last_date) or (jump_id < self.last_jump_id - 1000):
                    self.reset_session(curr_date)
                
                self.last_date = curr_date
                self.last_jump_id = jump_id
                
                bid1 = jump_data['bid1']
                ask1 = jump_data['ask1']
                
                # 1. Restore kinetic sequence with high-precision (v2.3)
                impact_seq = []
                buy_v, sell_v = 0, 0
                for imp in jump_data.get('impacts', []):
                    q_imp = KineticImpact(
                        offset_ms=imp.get("ms", 1),
                        volume=imp.get("vol", 1),
                        is_buy=imp.get("buy", True),
                        intensity=imp.get("intensity", 100.0),
                        offset_ns=imp.get("ns") # High-precision ns offset
                    )
                    impact_seq.append(q_imp)
                    if q_imp.is_buy: buy_v += q_imp.volume
                    else: sell_v += q_imp.volume

                engine_state = QuantumState(
                    jump_id=jump_id, bid1=bid1, ask1=ask1,
                    bid_vol1=jump_data['bid_vol1'], ask_vol1=jump_data['ask_vol1'],
                    total_t=0.0, buy_vol=buy_v, sell_vol=sell_v, 
                    duration_ms=jump_data.get('duration_ms', 1)
                )
                
                # 2. Engine computation (v2.3 Zero-Lag Architecture)
                # First, process with N=1 to get the current matrix for N-derivation
                res = self.engine.process_state(
                    engine_state, impact_seq, 
                    velocity_only=self.velocity_only,
                    horizon_n=1.0
                )
                current_matrix = res['matrix']
                horizon_n = calculate_physical_horizon(current_matrix, target_gain=self.target_gain)
                
                # Project probabilities for the derived horizon N
                # Use get_n_step_matrix to avoid double-processing the engine's internal counter
                powered_matrix = self.engine.matrix_builder.get_n_step_matrix(horizon_n)
                density = res['density']
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
                
                # Dynamic Threshold (Educational adaptation)
                dynamic_threshold = self.base_threshold
                
                # 4. Information Innovation Filter (v2.3 info_threshold)
                prob_diff = abs(up_prob - self.last_up_prob)
                self.last_up_prob = up_prob
                
                # 5. Signal Determination
                signal = 'NEUTRAL'
                is_warmup = jump_id < self.warmup_jumps
                
                if not is_warmup:
                    if up_prob > dynamic_threshold and prob_diff > self.info_threshold:
                        signal = 'LONG'
                    elif down_prob > dynamic_threshold and prob_diff > self.info_threshold:
                        signal = 'SHORT'
                
                # 6. Trading Simulation (Theoretical High-Parity Mode)
                # Exit positions
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
                
                # Entry positions
                if signal != 'NEUTRAL' and len(self.active_positions) < self.max_units:
                    direction = 1 if signal == 'LONG' else -1
                    # Simulation slippage (0.5 tick)
                    entry_price = (ask1 + 0.5 * self.tick_size) if direction == 1 else (bid1 - 0.5 * self.tick_size)
                    
                    self.active_positions.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'target_jump_id': jump_id + horizon_n
                    })
                    self.shared_state['total_trades'] += 1
                    if direction == 1: self.shared_state['long_count'] += 1
                    else: self.shared_state['short_count'] += 1

                # Calculate Unrealized PnL
                total_unrealized = 0.0
                for pos in self.active_positions:
                    curr_exit = bid1 if pos['direction'] == 1 else ask1
                    total_unrealized += pos['direction'] * (curr_exit - pos['entry_price'])

                # Performance Stats
                total_done = self.shared_state['wins'] + self.shared_state['losses']
                win_rate = (self.shared_state['wins'] / total_done * 100) if total_done > 0 else 0.0
                tunneling = (signal == 'LONG' and prob_dist[-1] > 0.3) or (signal == 'SHORT' and prob_dist[0] > 0.3)

                # 7. Atomic State Update
                self.shared_state.update({
                    'jump_id': jump_id,
                    'bid1': bid1,
                    'ask1': ask1,
                    'density': density,
                    'up_prob': up_prob,
                    'down_prob': down_prob,
                    'threshold': dynamic_threshold,
                    'signal': signal,
                    'win_rate': win_rate,
                    'unrealized_pnl': total_unrealized,
                    'active_units': len(self.active_positions),
                    'horizon_n': horizon_n,
                    'dimension': dim,
                    'tunneling_warning': tunneling,
                    'position': 'WARMING UP' if is_warmup else ('TRADING' if len(self.active_positions) > 0 else 'WAITING'),
                    'processed_events': self.shared_state['processed_events'] + len(jump_data.get('impacts', []))
                })

            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                import traceback
                traceback.print_exc()
                break

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
            
            # --- State Table ---
            table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
            table.add_row("Jump ID", f"[bold white]{state['jump_id']}[/]")
            table.add_row("Price", f"Ask: [bold red]{state['ask1']:.2f}[/] / Bid: [bold blue]{state['bid1']:.2f}[/]")
            table.add_row("Energy Density", f"{density_bar} {state['density']:.4f}")
            table.add_row("Hilbert Space", f"[bold magenta]{state.get('dimension', 5)}x{state.get('dimension', 5)}[/]")
            table.add_row("Horizon N", f"[bold yellow]{state['horizon_n']:.2f}[/] (Curvature)")
            table.add_row("UP Prob", up_bar)
            table.add_row("DOWN Prob", dn_bar)
            
            sig_text = f"[{sig_color}]{sig}[/]"
            if state['tunneling_warning']: sig_text += " [blink bold yellow]⚠️ TUNNELING[/]"
            table.add_row("ENGINE SIGNAL", sig_text)

            # --- Floating PnL Panel ---
            pnl = state['unrealized_pnl']
            pnl_color = "bold bright_green" if pnl >= 0 else "bold bright_red"
            pnl_panel = Panel(
                f"[{pnl_color}]{pnl:+.2f} pts[/]",
                title="[bold white]FLOATING PnL[/]",
                border_style=pnl_color,
                padding=(1, 4),
                subtitle=f"[dim]Total: {state['total_pnl']:+.2f}[/]"
            )

            # --- Performance Table ---
            perf_table = Table(box=box.SIMPLE, show_header=True, expand=True, border_style="dim")
            perf_table.add_column("Metrics", style="cyan")
            perf_table.add_column("Values", justify="right")
            perf_table.add_row("Active Units", f"[bold magenta]{state['active_units']}[/]")
            perf_table.add_row("Win Rate", f"[bold yellow]{state['win_rate']:.1f}%[/] ({state['wins']}W {state['losses']}L)")
            
            status_color = "yellow" if "WARMING" in state['position'] else "green"
            perf_table.add_row("Status", f"[bold {status_color}]{state['position']}[/]")

            # Assemble Dashboard
            top_group = Group(
                Panel(table, title="[bold blue]Market Quantum State[/]", border_style="blue"),
                pnl_panel
            )
            
            return Panel(
                Group(top_group, Panel(perf_table, title="[bold yellow]Performance Dashboard[/]", border_style="yellow")),
                title=f"[bold cyan]Demon Engine v2.3 (THEORY) (G:{self.target_gain} T:{self.base_threshold})[/]",
                subtitle="[dim]Quantum-Physical Market Analysis (Lecture Material Sync)[/]",
                border_style="bright_magenta",
                padding=(1, 2)
            )

        # Allow connection logs to print first
        await asyncio.sleep(1.0)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        with Live(generate_dashboard(), console=console, refresh_per_second=4) as live:
            while True:
                await asyncio.sleep(0.25)
                try:
                    live.update(generate_dashboard())
                except Exception:
                    pass

    async def run(self):
        await asyncio.gather(self.engine_loop(), self.ui_loop())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default="QQQ")
    parser.add_argument('--threshold', type=float, default=0.83)
    parser.add_argument('--gain', type=float, default=10.15)
    parser.add_argument('--vol-multiplier', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--v', action='store_true', help="Velocity Only Mode")
    args = parser.parse_args()
    
    predictor = QuantumPredictor(
        symbol=args.symbol, 
        base_threshold=args.threshold,
        target_gain=args.gain,
        vol_multiplier=args.vol_multiplier,
        warmup_jumps=args.warmup,
        velocity_only=args.v
    )
    try:
        asyncio.run(predictor.run())
    except KeyboardInterrupt:
        print("\nExiting.")

import sys
import os
import json
import logging
import asyncio
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
    def __init__(self, symbol="QQQ", host='127.0.0.1', port=9999, base_threshold=0.6):
        self.symbol = symbol
        self.host = host
        self.port = port
        self.engine = QuantumDemonEngine()
        self.last_date = None
        self.last_jump_id = -1
        self.base_threshold = base_threshold
        
        # Shared Memory (State for UI)
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
            'total_trades': 0,
            'long_count': 0,
            'short_count': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'position': 'NONE',
            'unrealized_pnl': 0.0,
            'jumps_left': 0,
            'horizon_n': 0
        }
        
        self.active_positions = [] 
        self.tick_size = 0.01
        
    def reset_session(self, new_date):
        """Reset the engine and statistics for a new trading session"""
        logger.warning(f"New session detected for date: {new_date}. Resetting state...")
        self.engine = QuantumDemonEngine()
        self.active_positions = []
        self.last_date = new_date
        
        # Reset UI statistics
        self.shared_state.update({
            'jump_id': 0,
            'density': 0.0,
            'up_prob': 0.0,
            'down_prob': 0.0,
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
            'position': 'NONE',
            'unrealized_pnl': 0.0,
            'jumps_left': 0,
            'horizon_n': 0
        })
        
    def _normalize_unitary(self):
        """Matrix normalization for numerical stability"""
        mat = self.engine.matrix_builder.matrix
        row_sums = mat.sum(axis=1)
        for i in range(mat.shape[0]):
            if row_sums[i] > 0:
                mat[i, :] = mat[i, :] / row_sums[i]
        self.engine.matrix_builder.matrix = mat

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
                
                # Check for session change (Date change or Jump ID drop)
                curr_date = jump_data.get('date')
                jump_id = jump_data['jump_id']
                
                if (self.last_date and curr_date != self.last_date) or (jump_id < self.last_jump_id - 1000):
                    self.reset_session(curr_date)
                
                self.last_date = curr_date
                self.last_jump_id = jump_id
                
                bid1 = jump_data['bid1']
                ask1 = jump_data['ask1']
                bid_vol1 = jump_data['bid_vol1']
                ask_vol1 = jump_data['ask_vol1']
                
                # 1. Restore kinetic sequence and energy
                impact_seq = []
                buy_v, sell_v = 0, 0
                for imp in jump_data.get('impacts', []):
                    q_imp = KineticImpact(
                        offset_ms=imp.get("ms", 1),
                        volume=imp.get("vol", 1),
                        is_buy=imp.get("buy", True),
                        intensity=imp.get("intensity", 100.0)
                    )
                    impact_seq.append(q_imp)
                    if q_imp.is_buy: buy_v += q_imp.volume
                    else: sell_v += q_imp.volume

                engine_state = QuantumState(
                    jump_id=jump_id, bid1=bid1, ask1=ask1,
                    bid_vol1=bid_vol1, ask_vol1=ask_vol1,
                    total_t=0.0, buy_vol=buy_v, sell_vol=sell_v, duration_ms=0
                )
                
                # 2. Engine computation and normalization
                res = self.engine.process_state(engine_state, impact_seq)
                self._normalize_unitary()
                
                # 3. Probability Calculation (Sync with Backtest logic: Apply Physical Horizon)
                density = res['density']
                q_v = max(1.0, res['quantized_V'])
                matrix = self.engine.matrix_builder.matrix
                
                # [Dynamic Physical Horizon Derivation]
                horizon_n = calculate_physical_horizon(matrix, target_gain=4.0)
                slippage = 1 if horizon_n > 200 else 3 # Accept higher friction (3 ticks) for short horizons
                
                dynamic_threshold = self.base_threshold + (0.2 / q_v)
                
                m_future = np.linalg.matrix_power(matrix, horizon_n)
                dim = matrix.shape[0]
                center_idx = dim // 2
                prob_dist = m_future[center_idx]
                
                up_prob = sum(prob_dist[:center_idx])
                down_prob = sum(prob_dist[center_idx+1:])
                
                # 4. Signal Determination
                signal = 'NEUTRAL'
                if up_prob > dynamic_threshold:
                    signal = 'LONG'
                elif down_prob > dynamic_threshold:
                    signal = 'SHORT'
                
                # 5. Trading Simulation (Multiple positions)
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
                
                # Calculate Unrealized PnL
                total_unrealized = 0.0
                for pos in self.active_positions:
                    curr_exit = bid1 if pos['direction'] == 1 else ask1
                    total_unrealized += pos['direction'] * (curr_exit - pos['entry_price'])

                # New Entry
                if signal != 'NEUTRAL':
                    slippage = 3 if density > 0.8 else 1
                    direction = 1 if signal == 'LONG' else -1
                    entry_price = (ask1 + (slippage * self.tick_size)) if direction == 1 else (bid1 - (slippage * self.tick_size))
                    
                    self.active_positions.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'target_jump_id': jump_id + horizon_n
                    })
                    self.shared_state['total_trades'] += 1
                    if direction == 1: self.shared_state['long_count'] += 1
                    else: self.shared_state['short_count'] += 1
                    self.shared_state['position'] = f"{signal} Entry"

                # Calculate Win Rate
                total_done = self.shared_state['wins'] + self.shared_state['losses']
                win_rate = (self.shared_state['wins'] / total_done * 100) if total_done > 0 else 0.0

                # Tunneling Detection
                tunneling = (signal == 'LONG' and prob_dist[-1] > 0.3) or (signal == 'SHORT' and prob_dist[0] > 0.3)

                # 6. Atomic State Update
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
                    'jumps_left': len(self.active_positions),
                    'horizon_n': horizon_n,
                    'dimension': res['dimension'],
                    'tunneling_warning': tunneling,
                    'processed_events': self.shared_state['processed_events'] + len(jump_data.get('impacts', []))
                })

            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
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
            table.add_row("Current Price", f"Ask: [bold red]{state['ask1']:.2f}[/] / Bid: [bold blue]{state['bid1']:.2f}[/]")
            table.add_row("Energy Density", f"{density_bar} {state['density']:.4f}")
            table.add_row("Quantum Dimension", f"[bold magenta]{state.get('dimension', 5)}x{state.get('dimension', 5)}[/] (Hilbert Space)")
            table.add_row("Horizon N", f"[bold yellow]{state['horizon_n']}[/] Jumps (Spacetime Curvature)")
            table.add_row("Base Threshold", f"[bold yellow]{self.base_threshold:.2f}[/]")
            table.add_row("Dynamic Barrier", f"{state['threshold']:.4f}")
            table.add_row("UP Prob", up_bar)
            table.add_row("DOWN Prob", dn_bar)
            
            sig_text = f"[{sig_color}]{sig}[/]"
            if state['tunneling_warning']: sig_text += " [blink bold yellow]⚠️ TUNNELING DETECTED[/]"
            table.add_row("ENGINE SIGNAL", sig_text)

            # --- Floating PnL Panel (Prominent) ---
            pnl = state['unrealized_pnl']
            pnl_color = "bold bright_green" if pnl >= 0 else "bold bright_red"
            pnl_panel = Panel(
                f"[{pnl_color}]{pnl:+.2f} pts[/]",
                title="[bold white]FLOATING PnL[/]",
                border_style=pnl_color,
                padding=(1, 4),
                subtitle=f"[dim]Total PnL: {state['total_pnl']:+.2f}[/]"
            )

            # --- Performance Table ---
            perf_table = Table(box=box.SIMPLE, show_header=True, expand=True, border_style="dim")
            perf_table.add_column("Metrics", style="cyan")
            perf_table.add_column("Values", justify="right")
            perf_table.add_row("Open Trades", f"[bold magenta]{state['jumps_left']}[/] (Quantum Superposition)")
            perf_table.add_row("Total Trades", f"{state['total_trades']} (L:{state['long_count']} S:{state['short_count']})")
            perf_table.add_row("Win Rate", f"[bold yellow]{state['win_rate']:.1f}%[/] ({state['wins']}W {state['losses']}L)")
            perf_table.add_row("Status", f"[bold white]{state['position']}[/]")

            # Assemble Dashboard
            top_group = Group(
                Panel(table, title="[bold blue]Market Quantum State[/]", border_style="blue"),
                pnl_panel
            )
            
            return Panel(
                Group(top_group, Panel(perf_table, title="[bold yellow]Performance Dashboard[/]", border_style="yellow")),
                title="[bold magenta]Money Maker Hybrid Engine v2.0[/]",
                subtitle="[dim]Quantum-Physical Market Analysis Interface[/]",
                border_style="bright_magenta",
                padding=(1, 2)
            )

        # Allow connection logs to print first, then clear the screen for a clean UI
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
    parser.add_argument('--symbol', type=str, default="QQQ", help="Symbol to predict (e.g., QQQ)")
    parser.add_argument('--threshold', type=float, default=0.6, help="Base entry threshold (default: 0.6)")
    args = parser.parse_args()
    
    predictor = QuantumPredictor(symbol=args.symbol, base_threshold=args.threshold)
    try:
        asyncio.run(predictor.run())
    except KeyboardInterrupt:
        print("\nExiting.")

import json
import argparse
import getpass
import time
from datetime import datetime
from binance_client import BinanceClient
import os

def load_signal(filepath='final_position.json'):
    """Load trading signal from JSON file"""
    try:
        if not os.path.exists(filepath):
            print(f"Signal file not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            signal = json.load(f)
            
        print(f"Loaded signal from {filepath}")
        print(f"Signal: {signal['symbol']} {signal['signal']} @ {signal['entry_price']}")
        return signal
    except Exception as e:
        print(f"Error loading signal: {str(e)}")
        return None

def execute_order(binance_client, signal, leverage=1):
    """Execute order based on trading signal with position conversion if needed"""
    try:
        # Symbol conversion process
        original_symbol = signal['symbol']
        print(f"Original symbol: {original_symbol}")
        
        # Convert to Binance futures format
        if original_symbol == "BTC-USD":
            symbol = "BTCUSDT"
        elif original_symbol == "ETH-USD":
            symbol = "ETHUSDT"
        else:
            symbol = original_symbol.replace('-', '')
            if 'USD' in symbol and not symbol.endswith('USDT'):
                symbol = symbol.replace('USD', 'USDT')
                
        print(f"Converted symbol for Binance: {symbol}")
        
        # 1. Check and cancel open orders
        check_and_cancel_open_orders(binance_client, symbol)
        
        # 2. Check current positions
        account_status = binance_client.get_account_status()
        print("Checking current positions...")
        
        current_position = None
        for position in account_status.get('positions', []):
            if position['symbol'] == symbol:
                current_position = position
                break
        
        # 3. Check entry price
        price = signal.get('entry_price', 0)
        if price <= 0:
            print(f"Invalid entry price in signal: {price}")
            return False
            
        # 4. Calculate order quantity (considering position conversion)
        quantity, signal_direction, position_direction = calculate_order_quantity(
            binance_client, symbol, signal, current_position, price
        )
        
        if quantity <= 0:
            print("Invalid order quantity. Cannot proceed.")
            return False
        
        # 5. Create entry order
        side = 'BUY' if signal_direction == "long" else 'SELL'
        entry_order_result = place_entry_order(
            binance_client, symbol, side, quantity, price, leverage
        )
        
        if not entry_order_result:
            print("Entry order failed. Aborting.")
            return False
            
        # 6. Wait for order to be processed
        print("Waiting for order to be processed...")
        time.sleep(2)
        
        # 7. Check positions again after order
        account_status = binance_client.get_account_status()
        
        updated_position = None
        for position in account_status.get('positions', []):
            if position['symbol'] == symbol:
                updated_position = position
                break
                
        # 8. Create exit order
        exit_order_result = place_exit_order(
            binance_client, symbol, signal, updated_position, leverage
        )
        
        # 9. Save results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(script_dir, 'order_executed.json')
        with open(json_file, 'w') as f:
            result_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'entry_order': entry_order_result,
                'exit_order': exit_order_result,
                'previous_position': {
                    'symbol': symbol,
                    'size': current_position['size'] if current_position else 0,
                    'direction': "long" if current_position and current_position['size'] > 0 else "short" if current_position else "none"
                } if current_position else None,
                'signal': signal,
                'position_converted': signal_direction != (position_direction if current_position else None)
            }
            
            if updated_position and abs(updated_position['size']) > 0:
                result_data['current_position'] = {
                    'symbol': symbol,
                    'size': updated_position['size'],
                    'entry_price': updated_position['entry_price'],
                    'direction': "long" if updated_position['size'] > 0 else "short"
                }
                
            json.dump(result_data, f, indent=4)
        
        return True
        
    except Exception as e:
        print(f"Error executing order: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
def place_limit_order(binance_client, signal_filepath='final_position.json', leverage=1):
    """Main function to load signal and place limit order"""
    # Load signal
    signal = load_signal(signal_filepath)
    if not signal:
        return False
        
    # Execute order based on signal
    result = execute_order(binance_client, signal, leverage)
    return result

def check_and_cancel_open_orders(binance_client, symbol):
    """Check and cancel unfilled orders"""
    try:
        print(f"Checking open orders for {symbol}...")
        open_orders = binance_client.get_open_orders(symbol)
        
        if open_orders:
            print(f"Found {len(open_orders)} open orders for {symbol}. Canceling all...")
            cancel_result = binance_client.cancel_all_orders(symbol)
            if cancel_result:
                print("All open orders canceled successfully")
            else:
                print("Failed to cancel some orders. Proceeding with caution.")
            
            # Wait for the cancellation to be reflected in the system
            print("Waiting for order cancellation to be processed...")
            time.sleep(2)
        else:
            print(f"No open orders found for {symbol}")
            
        return True
    except Exception as e:
        print(f"Error checking/canceling open orders: {str(e)}")
        return False
    
def calculate_order_quantity(binance_client, symbol, signal, current_position, price):
    """Calculate order quantity function (considering position conversion)"""
    try:
        # Get account information
        account = binance_client.get_account_balance()
        available_balance = account.get('availableBalance', 0)
        print(f"Available balance: {available_balance} USDT")
        
        # Check position direction
        signal_direction = "long" if signal['signal'].lower() == 'long' else "short"
        fraction = signal.get('fraction', 0.3)
        
        # Calculate base investment amount
        investment_amount = available_balance * fraction
        
        # Calculate simple quantity
        quantity = investment_amount / price
        print(f"Base quantity from available balance: {quantity}")
        
        # Check position direction and calculate additional quantity
        additional_quantity = 0
        position_direction = None
        
        if current_position and abs(current_position['size']) > 0:
            position_direction = "long" if current_position['size'] > 0 else "short"
            print(f"Found existing {position_direction} position: {current_position['size']} contracts @ {current_position['entry_price']}")
            
            # Compare signal direction with current position direction
            if signal_direction != position_direction:
                # If directions are opposite, position conversion is needed
                print(f"Position direction change needed: {position_direction} -> {signal_direction}")
                
                # Total position size + new order size
                additional_quantity = abs(current_position['size'])
                print(f"Adding {additional_quantity} to close existing position")
            else:
                # If directions match, maintain existing position
                print(f"Signal direction matches current position direction: {signal_direction}")
        
        # Add additional quantity if existing position is in opposite direction
        if additional_quantity > 0:
            quantity += additional_quantity
            print(f"Total quantity after adding position conversion: {quantity}")
        
        # Check minimum notional value (10 USDT minimum)
        if quantity * price < 10:
            print(f"Order notional value: {quantity * price} USDT is below minimum requirement")
            if quantity * price > 0:
                quantity = 10 / price  # Adjust quantity to meet minimum notional of 10 USDT
                print(f"Adjusted quantity to meet minimum notional: {quantity}")
            else:
                print("Order amount too small, below minimum notional value")
                return 0, signal_direction, position_direction
                
        return quantity, signal_direction, position_direction
    except Exception as e:
        print(f"Error calculating order quantity: {str(e)}")
        return 0, None, None
    
def place_entry_order(binance_client, symbol, side, quantity, price, leverage=1):
    """Create entry order function"""
    try:
        # Get symbol information
        symbol_info = binance_client.get_symbol_info(symbol)
        if symbol_info:
            print(f"Symbol info: {symbol_info}")
        
        # Place order - use round function for precision handling
        quantity_rounded = round(quantity, 3)  # Quantity with 3 decimal places
        price_rounded = round(price, 1)  # Price with 1 decimal place
        
        print(f"Placing {side} order: {quantity_rounded} {symbol} @ {price_rounded} USDT")
        print(f"Total order value: {quantity_rounded * price_rounded} USDT")
        
        # Margin type - use uppercase ('ISOLATED', 'CROSSED')
        order_result = binance_client.place_futures_order(
            symbol=symbol,
            side=side,
            quantity=quantity_rounded,
            order_type='LIMIT',
            price=price_rounded,
            leverage=leverage,
            margin_type='ISOLATED'  # Use uppercase
        )
        
        print(f"Order placed: {order_result}")
        return order_result
    except Exception as e:
        print(f"Error placing entry order: {str(e)}")
        return None

def place_exit_order(binance_client, symbol, signal, position_data, leverage=1):
    """Create conditional order according to exit strategy"""
    try:
        # Check for position existence
        if not position_data or 'size' not in position_data or abs(float(position_data.get('size', 0))) == 0:
            print(f"No open position found for {symbol}. Skipping exit order.")
            return None
            
        # Check position size and direction
        position_size = float(position_data.get('size', 0))
        position = "long" if position_size > 0 else "short"
        
        print(f"Found active {position} position: {position_size} contracts")
            
        # Get exit strategy information
        exit_strategy = signal.get('exit_strategy', {}).get(position)
        if not exit_strategy:
            print(f"No exit strategy defined for {position} position")
            return None
        
        # Calculate exit quantity
        fraction = exit_strategy.get('quantity_fraction', 0.3)
        exit_quantity = abs(position_size) * fraction
        exit_quantity_rounded = round(exit_quantity, 3)
        
        # Stop price
        stop_price = exit_strategy.get('stop_price')
        
        # Check current market price
        current_price = binance_client.get_current_price(symbol)
        print(f"Current market price: {current_price}")
        
        # Adjust stop price (based on market price)
        if position == "long":
            # Long position: stop must be set lower than current price
            adjusted_stop = min(stop_price, current_price * 0.99)  # 99% of current price
        else:
            # Short position: stop must be set higher than current price
            adjusted_stop = max(stop_price, current_price * 1.01)  # 101% of current price
            
        stop_price_rounded = round(adjusted_stop, 1)
        
        # Exit order direction
        side = exit_strategy.get('side')
        
        print(f"Placing STOP_MARKET {side} order: {exit_quantity_rounded} {symbol} @ {stop_price_rounded} USDT (adjusted from {stop_price})")
        
        # Get leverage and margin type information from position
        position_leverage = position_data.get('leverage', leverage)
        position_margin_type = position_data.get('margin_type', 'ISOLATED')
        
        # Replace with regular limit order
        try:
            exit_order_result = binance_client.place_futures_order(
                symbol=symbol,
                side=side,
                quantity=exit_quantity_rounded,
                order_type='STOP_MARKET',
                price=None,
                stop_price=stop_price_rounded,
                leverage=position_leverage,
                margin_type=position_margin_type
            )
            
            print(f"Stop Market order placed: {exit_order_result}")
            return exit_order_result
            
        except Exception as stop_error:
            # If stop market order fails, use regular limit order instead
            if "Order would immediately trigger" in str(stop_error):
                print("Stop Market order would immediately trigger. Placing regular limit order instead.")
                
                # Set limit price better than entry price
                entry_price = float(position_data.get('entry_price', 0))
                if position == "long":
                    # Long position: sell slightly higher than current price
                    limit_price = max(current_price * 1.005, entry_price * 1.01)
                else:
                    # Short position: buy slightly lower than current price
                    limit_price = min(current_price * 0.995, entry_price * 0.99)
                    
                limit_price_rounded = round(limit_price, 1)
                
                print(f"Placing LIMIT {side} order: {exit_quantity_rounded} {symbol} @ {limit_price_rounded} USDT")
                
                limit_order_result = binance_client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    quantity=exit_quantity_rounded,
                    order_type='LIMIT',
                    price=limit_price_rounded,
                    leverage=position_leverage,
                    margin_type=position_margin_type
                )
                
                print(f"Limit order placed: {limit_order_result}")
                return limit_order_result
            else:
                raise
        
    except Exception as e:
        print(f"Error placing exit order: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Binance Order Execution')
    parser.add_argument('--api-key', type=str, help='Binance API Key')
    parser.add_argument('--testnet', type=int, choices=[0, 1], default=1, help='Use testnet or not (0=Real, 1=Testnet)')
    parser.add_argument('--signal-file', type=str, default='final_position.json', help='Signal JSON file path')
    parser.add_argument('--leverage', type=int, default=1, help='Leverage for futures order')
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key if args.api_key else input("Binance API Key: ")
    api_secret = getpass.getpass("Binance Secret Key: ")

    # Determine if testnet should be used
    testnet = bool(args.testnet)
    env_name = 'TESTNET' if testnet else 'REAL TRADING'
    
    print(f"Connecting to Binance in {env_name} mode.")
    print(f"Using signal file: {args.signal_file}")
    print(f"Using leverage: {args.leverage}x")
    
    confirm = input(f"Confirm order placement on {env_name}? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled by user")
        exit()
    
    try:
        # Initialize Binance client
        binance_client = BinanceClient(api_key, api_secret, testnet)
        
        # Check connection
        account = binance_client.get_account_balance()
        if account:
            print(f"Total USDT Balance: {account.get('totalWalletBalance', 0)}")
            print(f"Available USDT Balance: {account.get('availableBalance', 0)}")
            print(f"Position Margin: {account.get('positionInitialMargin', 0)}")
            print(f"Unrealized Profit: {account.get('unrealizedProfit', 0)}")
            
            # Proceed with available balance
            available_balance = account.get('availableBalance', 0)
            if available_balance > 0:
                print(f"Using available balance: {available_balance} USDT for trading")
            else:
                print("Insufficient balance to place orders")
        else:
            print("Failed to retrieve account balance information")
            
        # Place order based on signal
        success = place_limit_order(binance_client, args.signal_file, args.leverage)
        
        if success:
            print("Order placement successful!")
        else:
            print("Order placement failed.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
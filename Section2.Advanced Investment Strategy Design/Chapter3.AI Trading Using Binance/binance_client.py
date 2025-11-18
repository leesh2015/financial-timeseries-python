import ccxt
from typing import Dict, Optional

class BinanceClient:
    def __init__(self, api_key: str = '', api_secret: str = '', testnet: bool = True):
        """Initialize Binance client"""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True  # Adjust for time difference automatically
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Sync time with Binance server
        self.exchange.load_time_difference()
        
        self.ws_thread = None

    def get_account_balance(self) -> dict:
        """Fetch futures account balance from Binance"""
        try:
            # Add recvWindow and timestamp parameters first
            params = {
                'recvWindow': 5000,
                'timestamp': self.exchange.milliseconds()
            }
                
            account_info = self.exchange.fapiPrivateV2GetAccount(params)
            
            # Extract USDT balance info
            assets = account_info.get('assets', [])
            usdt_asset = next((asset for asset in assets if asset['asset'] == 'USDT'), None)
            
            if usdt_asset:
                return {
                    'totalWalletBalance': float(usdt_asset['walletBalance']),
                    'availableBalance': float(usdt_asset['availableBalance']),
                    'positionInitialMargin': float(usdt_asset['initialMargin']),
                    'unrealizedProfit': float(usdt_asset['unrealizedProfit'])
                }
            return {}
            
        except Exception as e:
            return {}

    def place_futures_order(self, symbol: str, side: str, quantity: float, 
                           order_type: str = 'MARKET', price: float = None,
                           leverage: int = 1, margin_type: str = 'ISOLATED',
                           stop_price: float = None) -> dict:
        """Place a futures order with the specified parameters"""
        try:
            # Add timestamp parameter
            params = {
                'timestamp': self.exchange.milliseconds(),
                'recvWindow': 5000
            }

            # Set leverage
            leverage_response = self.exchange.fapiPrivatePostLeverage({
                'symbol': symbol,
                'leverage': leverage,
                **params  # Include timestamp
            })
            print(f"Leverage set to {leverage}x: {leverage_response}")
            
            # Set margin type
            try:
                margin_response = self.exchange.fapiPrivatePostMarginType({
                    'symbol': symbol,
                    'marginType': margin_type,
                    **params  # Include timestamp
                })
                print(f"Margin type set: {margin_response}")
            except Exception as margin_error:
                if '"code":-4046' not in str(margin_error):
                    raise margin_error

            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type,
                **params  # Include timestamp
            }

            # Validate order parameters
            if order_type == 'LIMIT':
                if not price or price <= 0:
                    raise ValueError("Invalid price for limit order")
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'
            elif order_type == 'STOP_MARKET':
                if not stop_price or stop_price <= 0:
                    raise ValueError("Invalid stop price for stop market order")
                order_params['stopPrice'] = stop_price

            # Place the order
            response = self.exchange.fapiPrivatePostOrder(order_params)

            # Combine order response with leverage and margin info
            full_response = {
                **response,
                'leverage': leverage_response.get('leverage', leverage),
                'maxNotionalValue': leverage_response.get('maxNotionalValue'),
                'marginType': margin_type
            }

            return full_response

        except Exception as e:
            print(f"Error placing futures order: {str(e)}")
            raise

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol"""
        try:
            return self.exchange.fapiPrivate_post_leverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })
        except Exception as e:
            print(f"Error setting leverage: {str(e)}")
            return {}

    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"Error getting current price: {e}")
            # Fallback to mark price if ticker fetch fails
            try:
                mark_price = self.get_mark_price(symbol)
                return mark_price
            except:
                raise Exception(f"Failed to get price for {symbol}")

    def get_current_orderbook(self, symbol: str) -> Optional[dict]:
        """Get latest orderbook for symbol"""
        return self.orderbook_data.get(symbol.replace('/', '').upper())

    def get_mark_price(self, symbol: str) -> float:
        """
        Get mark price for a futures symbol
        :param symbol: Trading symbol (e.g. 'BTCUSDT')
        :return: Current mark price as float
        """
        try:
            # Use futures mark price endpoint
            response = self.exchange.fapiPublicGetPremiumIndex({'symbol': symbol})
            return float(response['markPrice'])
        except Exception as e:
            try:
                # Fallback to bookTicker price if WebSocket is running
                if symbol in self.current_prices:
                    return self.current_prices[symbol]
            except:
                pass
            raise Exception(f"Failed to get mark price for {symbol}")

    def get_account_status(self) -> dict:
        try:
            # Add timestamp and recvWindow parameters
            params = {
                'timestamp': self.exchange.milliseconds(),
                'recvWindow': 5000
            }

            # Get account info with timestamp parameter
            account = self.exchange.fapiPrivateV2GetAccount(params)
            
            # Initialize status dictionary
            status = {
                'balance': {},
                'positions': [],
                'leverage': {},
                'unrealized_pnl': 0.0
            }
            
            for asset in account.get('assets', []):
                if asset['asset'] == 'USDT':
                    status['balance'] = {
                        'wallet_balance': float(asset.get('walletBalance', 0)),
                        'unrealized_profit': float(asset.get('unrealizedProfit', 0)),
                        'available_balance': float(asset.get('availableBalance', 0)),
                        'margin_balance': float(asset.get('marginBalance', 0)),
                        'position_margin': float(asset.get('initialMargin', 0)),
                        'order_margin': float(asset.get('openOrderInitialMargin', 0))
                    }

            positions = account.get('positions', [])
            for pos in positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:  # Only include open positions
                    position_info = {
                        'symbol': pos['symbol'],
                        'size': position_amt,
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedProfit', 0)),
                        'leverage': int(pos.get('leverage', 1)),
                        'margin_type': 'ISOLATED' if pos.get('isolated', False) else 'CROSS',
                        'notional': float(pos.get('notional', 0)),
                        'isolated_wallet': float(pos.get('isolatedWallet', 0)) if pos.get('isolated', False) else 0
                    }
                    
                    status['positions'].append(position_info)
                    status['leverage'][pos['symbol']] = position_info['leverage']
                    status['unrealized_pnl'] += position_info['unrealized_pnl']

            return status
                
        except Exception as e:
            return {}

    def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol information including precision"""
        try:
            exchange_info = self.exchange.fapiPublicGetExchangeInfo()
            for sym in exchange_info['symbols']:
                if sym['symbol'] == symbol:
                    return {
                        'quantityPrecision': sym['quantityPrecision'],
                        'pricePrecision': sym['pricePrecision'],
                        'minQty': float(sym['filters'][1]['minQty']),
                        'maxQty': float(sym['filters'][1]['maxQty'])
                    }
            return None
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return None
            
    def get_open_orders(self, symbol=None):
        """Get all open (unfilled) orders for a symbol or all symbols"""
        try:
            params = {
                'timestamp': self.exchange.milliseconds(),
                'recvWindow': 5000
            }
            
            if symbol:
                params['symbol'] = symbol
                
            open_orders = self.exchange.fapiPrivateGetOpenOrders(params)
            return open_orders
        except Exception as e:
            print(f"Error getting open orders: {str(e)}")
            return []
            
    def cancel_all_orders(self, symbol=None):
        """Cancel all open orders for a symbol or all symbols"""
        try:
            params = {
                'timestamp': self.exchange.milliseconds(),
                'recvWindow': 5000
            }
            
            if symbol:
                params['symbol'] = symbol
                result = self.exchange.fapiPrivateDeleteAllOpenOrders(params)
                print(f"Canceled all open orders for {symbol}: {result}")
            else:
                # Get all symbols with open orders
                open_orders = self.get_open_orders()
                symbols = set([order['symbol'] for order in open_orders])
                
                # Cancel orders for each symbol
                for sym in symbols:
                    params['symbol'] = sym
                    result = self.exchange.fapiPrivateDeleteAllOpenOrders(params)
                    print(f"Canceled all open orders for {sym}: {result}")
                    
            return True
        except Exception as e:
            print(f"Error canceling orders: {str(e)}")
            return False
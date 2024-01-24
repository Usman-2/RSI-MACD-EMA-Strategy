import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Function to download historical stock data
def download_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# Function to count buy signals, losses, and calculate win rate
def count_buy_sell_losses(predictions, actual_prices, threshold_percent=2):
    buy_signals = 0
    losses = 0

    for i in range(len(predictions)):
        if predictions[i] == 1:
            buy_signals += 1
            if i + 1 < len(actual_prices):
                if (actual_prices.iloc[i + 1] / actual_prices.iloc[i] - 1) < -threshold_percent / 100:
                    losses += 1

    win_rate = (buy_signals - losses) / buy_signals * 100 if buy_signals > 0 else 0
    return losses, win_rate

# Function to execute buy orders based on predictions
def execute_buy_orders(predictions, open_prices, close_prices, sell_within_days=365):
    buy_orders = []
    cumulative_pnl = 0
    signals_df = pd.DataFrame()

    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            buy_price = open_prices.iloc[i]
            print(f"Buy signal at index {i}, Buy Price: {buy_price}")  # Debugging print

            sell_price = None
            pnl = 0
            sold = False

            # Find the next sell signal within the specified window
            for j in range(i+1, min(i+sell_within_days+1, len(predictions))):
                print(f"Checking for sell signal at index {j}")  # Debugging print

                if predictions[j] == -1:  # Sell signal
                    sell_price = close_prices.iloc[j]  # Use closing price for selling
                    pnl = sell_price - buy_price
                    cumulative_pnl += pnl
                    sold = True

                    print(f"Sell signal found at index {j}, Sell Price: {sell_price}, PnL: {pnl}")  # Debugging print
                    break

            if not sold:  # If not sold, log the info
                print(f"No sell signal found within {sell_within_days} days for buy at index {i}")  # Debugging print

            order_details = {
                'Date': open_prices.index[i],
                'Buy_Price': buy_price,
                'Sell_Price': sell_price if sold else 'Not Sold',
                'PnL': pnl,
                'Cumulative_PnL': cumulative_pnl
            }
            buy_orders.append(order_details)
            signals_df = pd.concat([signals_df, pd.DataFrame([order_details])], ignore_index=True)

    return buy_orders, signals_df


# Main script
tesla_symbol = 'TSLA'
tesla_start_date = '2023-01-01'
tesla_end_date = '2024-02-01'

# Download and process Tesla stock data
tesla_data = download_stock_data(tesla_symbol, tesla_start_date, tesla_end_date)
if tesla_data.empty:
    raise Exception("Failed to download Tesla data")

# Add technical indicators
tesla_data['ema'] = ta.trend.ema_indicator(tesla_data['Close'], window=14)
tesla_data['rsi'] = ta.momentum.rsi(tesla_data['Close'], window=14)
tesla_data['macd'] = ta.trend.macd(tesla_data['Close'], window_fast=12, window_slow=26)
tesla_data['vwap'] = ta.volume.volume_weighted_average_price(tesla_data['High'], tesla_data['Low'], tesla_data['Close'], tesla_data['Volume'])
tesla_data['Signal'] = 0
tesla_data.loc[tesla_data['ema'] > tesla_data['macd'], 'Signal'] = 1  # Buy signal
tesla_data.loc[tesla_data['ema'] < tesla_data['macd'], 'Signal'] = -1  # Sell signal
tesla_data.dropna(inplace=True)

# Prepare features and labels
X_tesla = tesla_data[['ema', 'rsi', 'macd', 'vwap']]
y_tesla = tesla_data['Signal']

# Split data for training and testing
X_train_tesla, X_test_tesla, y_train_tesla, y_test_tesla = train_test_split(X_tesla, y_tesla, test_size=0.2, random_state=42)

# Train the model
model_tesla = DecisionTreeClassifier(random_state=42)
model_tesla.fit(X_train_tesla, y_train_tesla)

# Prepare new data for prediction
new_tesla_data = download_stock_data(tesla_symbol, tesla_start_date, tesla_end_date)
if new_tesla_data.empty:
    raise Exception("Failed to download new Tesla data")

# Add indicators to new data
new_tesla_data['ema'] = ta.trend.ema_indicator(new_tesla_data['Close'], window=14)
new_tesla_data['rsi'] = ta.momentum.rsi(new_tesla_data['Close'], window=14)
new_tesla_data['macd'] = ta.trend.macd(new_tesla_data['Close'], window_fast=12, window_slow=26)
new_tesla_data['vwap'] = ta.volume.volume_weighted_average_price(new_tesla_data['High'], new_tesla_data['Low'], new_tesla_data['Close'], new_tesla_data['Volume'])
new_tesla_data.dropna(inplace=True)
new_tesla_predictions = model_tesla.predict(new_tesla_data[['ema', 'rsi', 'macd', 'vwap']])

# Execute buy orders
buy_orders, buy_signals_df = execute_buy_orders(new_tesla_predictions, new_tesla_data['Open'], new_tesla_data['Close'])

# Count losses and win rate
losses, win_rate = count_buy_sell_losses(new_tesla_predictions, new_tesla_data['Close'])
print(f'Number of Buy Signals: {len(buy_orders)}')
print(f'Number of Losses: {losses}')
print(f'Win Rate: {win_rate:.2f}%')

# Save to CSV
buy_signals_df.to_csv('buy_signals.csv', index=False)

# Final cumulative PnL
final_cumulative_pnl = buy_signals_df['Cumulative_PnL'].iloc[-1]
print(f'Final Cumulative P&L: {final_cumulative_pnl}')

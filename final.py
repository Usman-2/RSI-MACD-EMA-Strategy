import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to download historical stock data
def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')
    return data

# Function to count buy/sell signals and losses
def count_buy_sell_losses(predictions, actual_prices, threshold_percent=2):
    buy_signals = 0
    sell_signals = 0
    losses = 0
    
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            buy_signals += 1
            if i + 1 < len(actual_prices):  # Ensure we are not at the last data point
                if (actual_prices.iloc[i + 1] / actual_prices.iloc[i] - 1) < -threshold_percent / 100:
                    losses += 1
        elif predictions[i] == -1:  # Sell signal
            sell_signals += 1
            
    return buy_signals, sell_signals, losses

# Function to execute buy/sell orders based on signals
def execute_orders(predictions, open_prices, current_position):
    orders = []
    closed_signals = []

    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            if current_position == 'none':
                orders.append(('Buy', open_prices.iloc[i]))
                current_position = 'long'
            elif current_position == 'short':
                closed_signals.append(('Closed', open_prices.iloc[i], current_position))
                orders.append(('Buy', open_prices.iloc[i]))
                current_position = 'long'
        elif predictions[i] == -1:  # Sell signal
            if current_position == 'long':
                orders.append(('Sell', open_prices.iloc[i]))
                closed_signals.append(('Closed', open_prices.iloc[i], current_position))
                current_position = 'none'
            elif current_position == 'none':
                orders.append(('Sell', open_prices.iloc[i]))
                current_position = 'short'

    if current_position != 'none':
        closed_signals.append(('Closed', open_prices.iloc[-1], current_position))

    print('Closed signals:', closed_signals)  # Print closed signals explicitly
    return orders, closed_signals, current_position

# Define the stock symbol and timeframe for Tesla
tesla_symbol = 'GOOGL'
tesla_start_date = '2022-02-02'
tesla_end_date = '2024-01-01'

# Download historical data for Tesla
tesla_data = download_stock_data(tesla_symbol, tesla_start_date, tesla_end_date)

# Add technical indicators
tesla_data['ema'] = ta.trend.ema_indicator(tesla_data['Close'], window=14)
tesla_data['rsi'] = ta.momentum.rsi(tesla_data['Close'], window=14)
tesla_data['macd'] = ta.trend.macd(tesla_data['Close'], window_fast=12, window_slow=26)

# Add VWAP indicator
tesla_data['vwap'] = ta.volume.volume_weighted_average_price(tesla_data['High'], tesla_data['Low'], tesla_data['Close'], tesla_data['Volume'])

# Create buy/sell signals based on your strategy
tesla_data['Signal'] = 0
# Replace the following lines with your specific strategy
# For example, you can use the crossover of ema and macd for signals
tesla_data.loc[tesla_data['ema'] > tesla_data['macd'], 'Signal'] = 1  # Buy signal
tesla_data.loc[tesla_data['ema'] < tesla_data['macd'], 'Signal'] = -1  # Sell signal

# Drop NaN values and prepare features (X) and labels (y)
tesla_data.dropna(inplace=True)  # Handle missing values by dropping them
X_tesla = tesla_data[['ema', 'rsi', 'macd', 'vwap']]
y_tesla = tesla_data['Signal']

# Split the dataset into training and testing sets
X_train_tesla, X_test_tesla, y_train_tesla, y_test_tesla = train_test_split(X_tesla, y_tesla, test_size=0.2, random_state=42)

# Create and train a decision tree classifier for Tesla
model_tesla = DecisionTreeClassifier(random_state=42)
model_tesla.fit(X_train_tesla, y_train_tesla)

# Make predictions on the test set for Tesla
predictions_tesla = model_tesla.predict(X_test_tesla)

# Calculate accuracy for Tesla
accuracy_tesla = accuracy_score(y_test_tesla, predictions_tesla)
print(f'Tesla Accuracy: {accuracy_tesla}')

# Now, you can use the trained model for predictions on new data for Tesla
# Replace 'new_tesla_data' with your new data for prediction
new_tesla_data = yf.download(tesla_symbol, start=tesla_start_date, end=tesla_end_date, interval='1h')  # Your new data for Tesla
new_tesla_data['ema'] = ta.trend.ema_indicator(new_tesla_data['Close'], window=14)
new_tesla_data['rsi'] = ta.momentum.rsi(new_tesla_data['Close'], window=14)
new_tesla_data['macd'] = ta.trend.macd(new_tesla_data['Close'], window_fast=12, window_slow=26)
new_tesla_data['vwap'] = ta.volume.volume_weighted_average_price(new_tesla_data['High'], new_tesla_data['Low'], new_tesla_data['Close'], new_tesla_data['Volume'])
new_tesla_data.dropna(inplace=True)  # Handle missing values by dropping them

# Initial current_position value
current_position = 'none'

# Execute buy/sell orders based on predictions and open prices
orders, closed_signals, current_position = execute_orders(predictions_tesla, new_tesla_data['Open'], current_position)

# Print buy/sell predictions for the new data for Tesla
print('Buy/Sell orders for new Tesla data:')
for order in orders:
    print(order)

# Print all closed positions
print('Closed Positions:')
for closed_signal in closed_signals:
    print(closed_signal)

# After making predictions on new data for Tesla
buy_signals, sell_signals, losses = count_buy_sell_losses(predictions_tesla, new_tesla_data['Close'])

# Append one more value to the closed positions variable
if current_position != 'none':
    closed_signals.append(('Closed', new_tesla_data['Close'].iloc[-1], current_position))

# Print the count of buy and sell signals, and losses
print(f'Number of Buy Signals: {buy_signals}')
print(f'Number of Sell Signals: {sell_signals}')
print(f'Number of Losses: {losses}')
print('Wins:', buy_signals - losses)
print('Win Rate:', (buy_signals - losses) / buy_signals * 100)

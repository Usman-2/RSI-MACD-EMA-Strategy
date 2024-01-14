import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to download historical stock data
def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
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
                if (actual_prices[i + 1] / actual_prices[i] - 1) < -threshold_percent / 100:
                    losses += 1
        elif predictions[i] == -1:  # Sell signal
            sell_signals += 1
            
    return buy_signals, sell_signals, losses

# Define the stock symbol and timeframe for Tesla
tesla_symbol = 'TSLA'
tesla_start_date = '2019-01-01'
tesla_end_date = '2024-01-01'

# Download historical data for Tesla
tesla_data = download_stock_data(tesla_symbol, tesla_start_date, tesla_end_date)

# Add technical indicators
tesla_data['ema'] = ta.trend.ema_indicator(tesla_data['Close'], window=14)
tesla_data['rsi'] = ta.momentum.rsi(tesla_data['Close'], window=14)
tesla_data['macd'] = ta.trend.macd(tesla_data['Close'], window_fast=12, window_slow=26)

# Create buy/sell signals based on your strategy
tesla_data['Signal'] = 0
# Replace the following lines with your specific strategy
# For example, you can use the crossover of ema and macd for signals
tesla_data.loc[tesla_data['ema'] > tesla_data['macd'], 'Signal'] = 1  # Buy signal
tesla_data.loc[tesla_data['ema'] < tesla_data['macd'], 'Signal'] = -1  # Sell signal

# Drop NaN values and prepare features (X) and labels (y)
tesla_data.dropna(inplace=True)  # Handle missing values by dropping them
X_tesla = tesla_data[['ema', 'rsi', 'macd']]
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
new_tesla_data = yf.download(tesla_symbol, start=tesla_start_date, end=tesla_end_date)  # Your new data for Tesla
new_tesla_data['ema'] = ta.trend.ema_indicator(new_tesla_data['Close'], window=14)
new_tesla_data['rsi'] = ta.momentum.rsi(new_tesla_data['Close'], window=14)
new_tesla_data['macd'] = ta.trend.macd(new_tesla_data['Close'], window_fast=12, window_slow=26)
new_tesla_data.dropna(inplace=True)  # Handle missing values by dropping them
new_tesla_predictions = model_tesla.predict(new_tesla_data[['ema', 'rsi', 'macd']])

# Print buy/sell predictions for the new data for Tesla
print('Predictions for new Tesla data:')
print(new_tesla_predictions)

# After making predictions on new data for Tesla
buy_signals, sell_signals, losses = count_buy_sell_losses(new_tesla_predictions, new_tesla_data['Close'])

# Print the count of buy and sell signals, and losses
print(f'Number of Buy Signals: {buy_signals}')
print(f'Number of Sell Signals: {sell_signals}')
print(f'Number of Losses: {losses}')
print('Wins:', buy_signals - losses)
print('Win Rate:', (buy_signals - losses) / buy_signals * 100)

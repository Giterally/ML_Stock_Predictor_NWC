import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the predictions from both models

# Predictions from Random Forest model (file_2)
rf_predictions = pd.read_csv('rf_predictions.csv')
rf_actual = rf_predictions['Target'].values
rf_predicted = rf_predictions['Predictions'].values

# Predictions from LSTM model (file_1)
lstm_predictions = pd.read_csv('lstm_predictions.csv')
lstm_actual = lstm_predictions['Actual'].values
lstm_predicted = lstm_predictions['Predicted'].values

def trading_simulation_rf(actual, predicted, initial_balance=10000):
    balance = initial_balance
    shares = 0
    balance_history = []

    for i in range(len(predicted)):
        if actual[i] == 0:  # Avoid divide by zero
            continue

        print(f"RF - Day {i}: Actual price = {actual[i]}, Predicted direction = {predicted[i]}, Current balance = {balance}, Shares = {shares}")

        if predicted[i] == 1:  # Buy signal
            shares_to_buy = balance // actual[i]
            if shares_to_buy > 0:
                balance -= shares_to_buy * actual[i]
                shares += shares_to_buy
                print(f"RF - Buying {shares_to_buy} shares at price {actual[i]}")
        elif predicted[i] == 0 and shares > 0:  # Sell signal
            balance += shares * actual[i]
            print(f"RF - Selling {shares} shares at price {actual[i]}")
            shares = 0
        balance_history.append(balance + shares * actual[i])

    # Sell any remaining shares at the last price
    if shares > 0:
        balance += shares * actual[-1]
        print(f"RF - Selling remaining {shares} shares at final price {actual[-1]}")
        shares = 0

    return balance, balance_history

def trading_simulation_lstm(actual, predicted, initial_balance=10000):
    balance = initial_balance
    shares = 0
    balance_history = []

    for i in range(len(predicted)):
        if actual[i] == 0:  # Avoid divide by zero
            continue

        print(f"LSTM - Day {i}: Actual price = {actual[i]}, Predicted price = {predicted[i]}, Current balance = {balance}, Shares = {shares}")

        if predicted[i] > actual[i]:  # Buy signal
            shares_to_buy = balance // actual[i]
            if shares_to_buy > 0:
                balance -= shares_to_buy * actual[i]
                shares += shares_to_buy
                print(f"LSTM - Buying {shares_to_buy} shares at price {actual[i]}")
        elif predicted[i] < actual[i] and shares > 0:  # Sell signal
            balance += shares * actual[i]
            print(f"LSTM - Selling {shares} shares at price {actual[i]}")
            shares = 0
        balance_history.append(balance + shares * actual[i])

    # Sell any remaining shares at the last price
    if shares > 0:
        balance += shares * actual[-1]
        print(f"LSTM - Selling remaining {shares} shares at final price {actual[-1]}")
        shares = 0

    return balance, balance_history

# Run trading simulation for Random Forest model
rf_final_balance, rf_balance_history = trading_simulation_rf(rf_actual, rf_predicted)

# Run trading simulation for LSTM model
lstm_final_balance, lstm_balance_history = trading_simulation_lstm(lstm_actual, lstm_predicted)

# Print the final balance for both models
print(f'Random Forest Final Balance: ${rf_final_balance:.2f}')
print(f'LSTM Final Balance: ${lstm_final_balance:.2f}')

# Plot the balance history for comparison
plt.figure(figsize=(14, 7))

plt.plot(rf_balance_history, label='Random Forest Balance')
plt.plot(lstm_balance_history, label='LSTM Balance')
plt.title('Trading Simulation Balance Over Time')
plt.xlabel('Days')
plt.ylabel('Balance')
plt.legend()

plt.tight_layout()
plt.show()

import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# OLD DATA LOADING FORMAT
# Load or download S&P 500 data
'''
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

# Create a timezone-aware datetime object
start_date = datetime(1990, 1, 1, tzinfo=pytz.UTC)
sp500 = sp500.loc[sp500.index >= start_date].copy()
'''

file_path = r"C:\Users\zcemrpo\OneDrive - University College London\Downloads\CODE\pytorch\AMZN.csv"
# Daily stock history of amazon
amzn = pd.read_csv(file_path)

# Plot S&P 500 closing prices
plt.figure(figsize=(12, 6))
amzn["Close"].plot.line(use_index=True)
plt.title("AMZN Closing Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Data preprocessing
amzn = amzn.drop(["Dividends", "Stock Splits"], axis=1, errors='ignore')
amzn["Tomorrow"] = amzn["Close"].shift(-1)
amzn["Target"] = (amzn["Tomorrow"] > amzn["Close"]).astype(int)

# Define prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Define backtesting function
def backtest(data, model, predictors, train_size=0.93):
    train_end = int(len(data) * train_size)
    train = data.iloc[:train_end].copy()
    test = data.iloc[train_end:].copy()
    predictions = predict(train, test, predictors, model)
    return predictions

# Create new predictors
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = amzn.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    amzn[ratio_column] = amzn["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    amzn[trend_column] = amzn.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

amzn = amzn.dropna(subset=amzn.columns[amzn.columns != "Tomorrow"])

# Create and train the model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Backtest with new predictors
predictions = backtest(amzn, model, new_predictors)

# Print results
print(predictions["Predictions"].value_counts())
print(f"Precision Score: {precision_score(predictions['Target'], predictions['Predictions'])}")
print(predictions["Target"].value_counts() / predictions.shape[0])


# Plot predictions
plt.figure(figsize=(12, 6))
predictions.plot()
plt.title("AMZN Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Target / Prediction")
plt.legend(["Target", "Predictions"])
plt.show()

# Display the final predictions
print(predictions)

# Save predictions
predictions.to_csv('rf_predictions.csv', index=False)

#predict 1 = price goes up, predict 0 = price goes

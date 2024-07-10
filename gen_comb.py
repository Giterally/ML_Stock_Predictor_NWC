import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

'''

OLD DATA LOADING FORMAT

# Load or download S&P 500 data
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
#daily stock history of amazon
amzn = pd.read_csv(file_path)

# Plot S&P 500 closing prices
plt.figure(figsize=(12, 6))
amzn["Close"].plot.line(use_index=True)
plt.title("AMZN Closing Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

#Data preprocessing
#this drops about 4 years of data so that we can calculate the Close_Ratio_1000 and Trend_1000 values completely
amzn = amzn.drop(["Dividends", "Stock Splits"], axis=1, errors='ignore')
amzn["Tomorrow"] = amzn["Close"].shift(-1)
amzn["Target"] = (amzn["Tomorrow"] > amzn["Close"]).astype(int)

# Define prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    #now, predict the probability that a row will be a 0 or 1
    preds = model.predict_proba(test[predictors])[:,1]
    #instead of threshold being 50%, the model will now trade based on a 60% threshold - this reduces the total number of trading days but increases the change that the price will go up/down when the model says it will.
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Define backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Create new predictors
#this will find the average if the market has gone up or down in the last 2, 5, 60 etc trading days 
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = amzn.rolling(horizon).mean()
    #It then compares it to the current closing price in a ratio
    ratio_column = f"Close_Ratio_{horizon}"
    amzn[ratio_column] = amzn["Close"] / rolling_averages["Close"]
    #for trend, we shift the sp500 by 1 because we can't include the current day - this results in leakage where data is being used to guess itself - model looks good on training data and bad off training data
    trend_column = f"Trend_{horizon}"
    amzn[trend_column] = amzn.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

amzn = amzn.dropna(subset=amzn.columns[amzn.columns != "Tomorrow"])

#Create and train the model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

#Backtest with new predictors
predictions = backtest(amzn, model, new_predictors)

#Print results
print(predictions["Predictions"].value_counts())
print(f"Precision Score: {precision_score(predictions['Target'], predictions['Predictions'])}")
print(predictions["Target"].value_counts() / predictions.shape[0])

#Plot predictions
plt.figure(figsize=(12, 6))
predictions.plot()
plt.title("AMZN Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Target / Prediction")
plt.legend(["Target", "Predictions"])
plt.show()

#Display the final predictions
print(predictions)


#now a higher % e.g. 55% correct

#things to do to extend the model
#some indexes e.g. on the other side of the world will trade outside of sp500 hours, so you could try to correlate their pricing to the sp500
#add in news, general macro conditions like interest rates and inflation
#add in key components of the sp500 like key stocks in key sectors e.g. if tech goes down, the index will go down e.g. 6 months later
#increasing reolution - daily data but could look at hourly, minute by minute, tick data IF YOU CAN GET IT (HARD) to make accurate predictions

#he said - can get very far just on this model


#END






#pull out the data
# Assuming predictions DataFrame contains 'Target' and 'Predictions' columns
predictions.to_csv('rf_predictions.csv', index=False)

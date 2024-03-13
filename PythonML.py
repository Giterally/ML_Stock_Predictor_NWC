#downloading S&P500 price data on every trading day
import yfinance as yf
import matplotlib.pyplot as plt

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

sp500.plot.line(y="Close", use_index=True)
#plt.show() #############################

del sp500['Dividends']
del sp500['Stock Splits']
#print(sp500)

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
#print(sp500) 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

#split the model into train and test data - last 100 used for test, rest used for train.
train = sp500.iloc[:-100]
test = sp500[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

RandomForestClassifier(min_samples_split=100, random_state=1)

from sklearn.metrics import precision_score
#get an array of predictions
preds = model.predict(test[predictors])

#turn array into a pandas series
import pandas as pd
preds = pd.Series(preds, index=test.index)

#calculate the precision score
precision = precision_score(test["Target"], preds)
#print("The precision score is: " + str(precision*100) + "%")

combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ["Target", "Prediction"]

combined.plot()
#plt.show() #############################


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
count = predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])
#52.8% accurate across around 6000 trading days

#check if that is good or not - find % of days where market actually went up
predictions["Target"].value_counts() / predictions.shape[0]
#this gives a slightly higher percentage around 53% so this machine is worse than just buying at the start of every day and selling at the end of every day


#adding more predictors to the model
horizons = [2, 5, 60, 250, 1000]
#this will find the average if the market has gone up or down in the last 2, 5, 60 etc trading days 

new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    #It then compares it to the current closing price in a ratio
    
    ratio_column = f"CLose_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    #for trend, we shift the sp500 by 1 because we can't include the current day - this results in leakage where data is being used to guess itself - model looks good on training data and bad off training data
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

#some rows will have 'NaN' because if python cannot compute the rolling average (only 4 values below, not 5 on 1990-01-08 so can't conpute Trend_5)
#drop these rows:
sp500 = sp500.dropna()
#this drops about 4 years of data so that we can calculate the Close_Ratio_1000 and Trend_1000 values completely


#update the model and change some parameters
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

#copy-paste & edit predict function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])

    #now, predict the probability that a row will be a 0 or 1
    preds = model.predict_proba(test[predictors])[:,1] #get second column: probability that the stock price will go up

    #instead of threshold being 50%, the model will now trade based on a 60% threshold - this reduces the total number of trading days but increases the change that the price will go up/down when the model says it will.
    preds[preds >= .6] = 1
    preds[preds < 6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])
#now a higher % e.g. 55% correct

#things to do to extend the model
#some indexes e.g. on the other side of the world will trade outside of sp500 hours, so you could try to correlate their pricing to the sp500
#add in news, general macro conditions like interest rates and inflation
#add in key components of the sp500 like key stocks in key sectors e.g. if tech goes down, the index will go down e.g. 6 months later
#increasing reolution - daily data but could look at hourly, minute by minute, tick data IF YOU CAN GET IT (HARD) to make accurate predictions

#can get very far just on this model
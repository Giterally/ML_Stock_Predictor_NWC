Explanation of the Two Stock Predictors



Model 1: Random Forest Classifier
Steps:

Data Loading:
Load historical stock prices for Amazon (AMZN) from a CSV file.

Data Preprocessing:
Calculate new features (rolling averages and trends over different horizons) and set the target variable for classification (whether the stock price will increase the next day).

Feature Engineering:
Create features based on rolling averages and trends over various periods (2, 5, 60, 250, 1000 days).

Model Training and Prediction:
Use a Random Forest Classifier to predict whether the stock price will go up or down.
Adjust the prediction threshold to 60% confidence for making a trading decision.

Backtesting:
Evaluate the model's performance on historical data by dividing it into training and testing sets iteratively.

Performance Evaluation:
Calculate and display the precision score and the distribution of predictions.



Model 2: Long Short-Term Memory (LSTM) Neural Network
Steps:

Data Loading:
Load historical stock prices for Amazon (AMZN) from a CSV file.

Data Preprocessing:
Prepare the data for time series prediction by creating lagged features (closing prices for the past 7 days).
Normalize the data using MinMaxScaler.

Dataset Preparation:
Convert the preprocessed data into PyTorch tensors and create datasets for training and testing.
Model Definition:
Define an LSTM model with one input feature (closing price), a hidden layer of size 4, and one output layer.
Training and Validation:
Train the LSTM model using the training dataset and validate its performance on the testing dataset.

Performance Evaluation:
Plot the actual and predicted stock prices to visually assess the model's performance.
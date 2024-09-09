With the help of various online tutorials and my own knowledge, I have built two Stock Predictors trained and tested on AMZN's daily pricing history. I have then compared output in a comparison of metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R²) value and developed a trading simulation in C++ for comparison of the efficacy of the 2 models. This was carried out after standardising the output to a .csv file containing 2 columns: "Target" and "Predictions" with ~300 rows of 1s and 0s.
A 0 indicated price decreasing that day and a 1 indicated increase. The "Target" was what actually happened, and the "Predictions" was what each model predicted.
This data was then fed into "comparison_script.py" for the error comparison and the "trading_simulator" directory which contains the trading simulator built using C++.


Model 1: Random Forest Classifier
Random Forest Models work by training multiple decision trees with randomized parameters and averaging their results, which makes them resistant to overfitting compared to single decision trees or neural networks. They are chosen for their interpretability, unlike neural networks, and are robust to varying data distributions, outperforming linear models like regression. Random Forests run relatively quickly compared to slower alternatives like Gradient Boosting Machines and can easily capture non-linear relationships, which is crucial in stock prediction where patterns like opening price don’t follow a simple linear trend. While models like SVMs can handle non-linearity, Random Forests are faster and more intuitive to tune.
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
RNN LSTMs are designed to handle sequential data by learning long-term dependencies, making them ideal for time series tasks like stock price prediction, where past data significantly influences future outcomes. Unlike feedforward neural networks that treat inputs independently, LSTMs process sequences and maintain memory through hidden states, allowing them to capture patterns across time. They are robust against noise due to their internal gating mechanisms and excel at modeling non-linear relationships, outperforming linear models like regression and basic RNNs. While simpler models like feedforward networks or linear regression struggle with temporal dependencies and non-linear trends, LSTMs are more suited for complex, real-world data sequences.
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



For comparison results, please view my github pages at: 'https://giterally.github.io/' 

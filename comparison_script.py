import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the predictions from both models
# Assuming file_1.py and file_2.py save their outputs as CSV files

# Predictions from Random Forest model (file_2)
rf_predictions_filepath = r"C:\Users\zcemrpo\OneDrive - University College London\Downloads\CODE\PyRFG\rf_predictions.csv"
rf_predictions = pd.read_csv('rf_predictions.csv')
rf_actual = rf_predictions['Target'].values
rf_predicted = rf_predictions['Predictions'].values

# Predictions from LSTM model (file_1)
lstm_predictions_filepath = r"C:\Users\zcemrpo\OneDrive - University College London\Downloads\CODE\PyRFG\lstm_predictions.csv"
lstm_predictions = pd.read_csv(lstm_predictions_filepath)
lstm_actual = lstm_predictions['Actual'].values
lstm_predicted = lstm_predictions['Predicted'].values

# Mean Absolute Error
mae_rf = mean_absolute_error(rf_actual, rf_predicted)
mae_lstm = mean_absolute_error(lstm_actual, lstm_predicted)

# Root Mean Squared Error
rmse_rf = np.sqrt(mean_squared_error(rf_actual, rf_predicted))
rmse_lstm = np.sqrt(mean_squared_error(lstm_actual, lstm_predicted))

# R-squared
r2_rf = r2_score(rf_actual, rf_predicted)
r2_lstm = r2_score(lstm_actual, lstm_predicted)

# Print the comparison results
print(f'Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}, R^2: {r2_rf}')
print(f'LSTM - MAE: {mae_lstm}, RMSE: {rmse_lstm}, R^2: {r2_lstm}')

# Plot the actual vs predicted values for comparison
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(rf_actual, label='Actual Close')
plt.plot(rf_predicted, label='Predicted Close')
plt.title('Random Forest Predictions vs Actual')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lstm_actual, label='Actual Close')
plt.plot(lstm_predicted, label='Predicted Close')
plt.title('LSTM Predictions vs Actual')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()

plt.tight_layout()
plt.show()

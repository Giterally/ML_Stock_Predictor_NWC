import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

file_path = r"C:\Users\zcemrpo\OneDrive - University College London\Downloads\CODE\pytorch\AMZN.csv"
#daily stock history of amazon
data = pd.read_csv(file_path)

#simplifying data to relevant columns
data = data[["Date", "Close"]]

#tells pytorch to use gpu (default use is cpu)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print(device)  this should give 'gpu' if using gpu or 'cpu' otherwise

#simple transformations
#make the date column a pandas datatype
data["Date"] = pd.to_datetime(data["Date"])

'''
#plot the data against close price
plt.plot(data["Date"], data["Close"])
plt.show()
'''

#now need to set up the dataframe in a way that's close to how
#the model is going to take the input and train off it.
#an LSTM is looking at history - for a particular date it wants to know the closing
#price at the data and the day before that and before that etc:

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps): #the number of steps is the 'lookback window'
    #make a deepcopy of the dataframe
    df = dc(df)
    #set the index of the dataframe to 'date'
    df.set_index("Date", inplace=True)
    #shifts the dataframe for the number of lookback windows
    for i in range(1, n_steps+1):
        df[f"Close(t-{i})"] = df["Close"].shift(i)

    df.dropna(inplace = True)

    return df
lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)

#print(shifted_df)

#the dateframe produced has the closing price
#for each date and then the 7 previous day closing prices
#This is useful because we now have an input and output pair:
#an input matrix x and an output vector y. 
#The input matrix is the previous 7 days of closing prices:
#"Close(t-1)" -> "Close(t-7)" and the output vector is the
#current day closing price "Close"
#the input matrix is used to predict the output vector
#we are trying to learn the sequential pattern which is why
#we are using a sequential-based model: rnn (recurrent neural networks)
#e.g. recurrent network reads going down, going down etc...
#and therefore predicts the price will go down

shifted_df_as_np = shifted_df.to_numpy()
#print(shited_df_as_np)

#just going to run a scaler on all of the data
from sklearn.preprocessing import MinMaxScaler

#the feature range puts all the values between -1 and 1
scaler = MinMaxScaler(feature_range = (-1, 1))
#run the transform and fit it onto the matrix
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
#print(shifted_df_as_np)

#now define an x and y
X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

#print(X.shape, y.shape)
#prints (6509, 7) and (6059, 0) so same number of rows but X has 7 different features

#mirror X.shape in the horizontal direction so it goes from the
#Close(t-7) row to the Close(t-1) row since for LSTMs we want it
#to recurringly get the most updated answer until it gets as
#close to the real Close value as possible
X = dc(np.flip(X, axis = 1))

#split into train and test data
#use the first 95% as train and the last 5% as test
split_index = int(len(X) * 0.95)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#prints ((6183, 7), (326, 7), (6183,), (326,))

#requirement for pytorch LSTMs to have an extra dimension at the
#end so need to do a bit of reshaping where the matricies and y
#vectors get another dimension at the end.

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#prints ((6183, 7, 1), (326, 7, 1), (6183, 1), (326, 1))

#this is all in numpy, but since we are using pytorch
#wrap this in pytorch tensors

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#prints (torch.Size([6183, 7, 1]), torch.Size([6183, 7, 1]), torch.Size([326, 7, 1]), torch.Size([6183, 1]), torch.Size([326, 1]))

#generally when training models in pytorch, you use datasets rather than raw tensors, so we are going to creat the dataset object:

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

#wrap the datasets in dataloaders to get batches

from torch.utils.data import DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#useful for visualisation
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break




class LSTM(nn.Module):
    #takes in input_size: number of features 1, hidden_size: however dimension we want to be in the middle, we want 4 num_stacked_layers: you can stack LSTMs because as they recurrently run through themselves, they produce a sequence of hidden layers, we are using 1 since the model is going to overfit pretty well anyway.
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        #after LSTM, it does the recurrent part, you want a fully connected layer which maaps from the hidden_size to 1, because at the end of the model we need it to predict just one value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)
model


def train_one_epoch():
    #set the model to training mode and print the epoch
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    #start to accumulate a running loss value
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        #comparing the model's output to the ground truth (real value) is the loss
        #the loss is a tensor with one value
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        #zero out the gradient
        optimizer.zero_grad()
        #do a backward pass through the loss to calculate the gradient
        loss.backward()
        #take a slight step in the direction of the gradient to make the model the model better
        optimizer.step()

        #for every 100 batches, get the average loss every 100 batches and print it out
        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    #model train is false, so in valuation mode
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            #get and accumulate loss again
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()




#the training loop 
learning_rate = 0.001
num_epochs = 10
#the loss function is the mean squared error since it is a from of regression problem - predicting a continuous value
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()





#code to do some plotting
with torch.no_grad():
    #put all the X_train stuff in to get predictions for the first 95% of the data, remember the model has already seen this stuff so it should do well
    predicted = model(X_train.to(device)).to('cpu').numpy()

'''
#plot y_train: the actual closing against the predicted closing
plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
'''


#for the abovve axis, the 'Close' values are between -1 and 1, so we need to do the opposite of what our transform did so that we can get out original scale back - the dollar values which are actually useful
train_predictions = predicted.flatten()
#it's all about making a dummy matrix with the same shape as what the scalar is used to
#the dummy matrix is used such that it's first column is our predictions and then apply the inverse transform
dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

#get back train predictions in the right scale
train_predictions = dc(dummies[:, 0])
#print(train_predictions) gives array([  0.3959165 ,   0.39559413,   0.39486046, ..., 172.31433271, 171.69293455, 171.57329039])

#for the ground truth / y_train, do the same thing
dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])
#print(new_y_train) gives array([7.91646265e-02, 7.65634249e-02, 7.52572660e-02, ..., 1.69091505e+02, 1.73315001e+02, 1.68871003e+02])


'''
#redo plot
plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
'''

#get the test predictions and convert to the proper scale
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])


dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()






#pull out the data
# Assuming new_y_test and test_predictions are the final outputs
lstm_results = pd.DataFrame({'Actual': new_y_test, 'Predicted': test_predictions})
lstm_results.to_csv('lstm_predictions.csv', index=False)

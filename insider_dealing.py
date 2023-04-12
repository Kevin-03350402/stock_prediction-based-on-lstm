import time
import datetime
import pandas as pd
import ssl
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import statistics
from random import randint
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
def historical_data(tinker):
    # target date range
    period1 = int(time.mktime(datetime.datetime(2001, 11, 22, 23, 27).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2023, 4, 11, 23, 27).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{tinker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    L = list(df["Close"])
    res = []
    for s in L:
        res.append(float(s))
    return res #return a list

def cut(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst)-n, 1):
        yield lst[i:i + n]

ticker = '000878.SZ'
# load the historical data 
price_info  = historical_data(ticker)

price_info = [x for x in price_info if str(x) != 'nan']

def differencing(prices):
    shadow = np.array(prices[:len(prices)-1])
    res = np.array(prices[1:len(prices)])
    return (100*np.divide((np.subtract(res , shadow)),shadow)).tolist()

price_info = differencing(price_info)


train_info = price_info[:int(len(price_info)*0.8)]
test_info = price_info[int(len(price_info)*0.2):]


print('Data loading done.')
obervation_period = 51

x_part_length = 50
y_part_length = obervation_period - x_part_length


noise_level = 0.05

train_info = train_info[:(len(train_info)//obervation_period)*obervation_period]

# set the observation period

# process the trainning set
raw_data_set = list(cut(train_info, obervation_period))

total_train_size = int(x_part_length*noise_level) + x_part_length

def fill_in_raw(raw_data_set,xp):
    X = []
    y = []
    for index in range(len(raw_data_set)):
        # split the rows 
        chunk = raw_data_set[index]
        X.append(chunk[:xp])
        y.append(chunk[xp:])
    return (np.array(X), np.array(y))

(X_train, y_train) = fill_in_raw(raw_data_set,x_part_length)
# due to the size of the data, the batch size can be at most 26



# process the test set
test_info = test_info[:(len(test_info)//(total_train_size+y_part_length))*(total_train_size+y_part_length)]
raw_test_set = list(cut(test_info, (total_train_size+y_part_length)))


(X_test,y_test)= fill_in_raw(raw_test_set,total_train_size)

batch_size = 16

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=total_train_size, hidden_size=obervation_period*5, num_layers=3, batch_first=True)
        self.dropout =  nn.Dropout(p=0.2)
        self.linear = nn.Linear(obervation_period*5, y_part_length)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters(),lr = 0.0001)
loss_fn = nn.MSELoss()

loss_list = []

test_losses_list = []

max_iter = 100
for epoch in range(max_iter):
    
    total_loss = 0
    n_batches = 0
    for batch, (X, y_batch) in enumerate(train_dataloader):
        # for each row, we add some random noise
        noise_x = []
        n_batches += 1
        
        for row_index in range(X.shape[0]):
            chunk = X[row_index,:]
            not_polluted_x = chunk.tolist()

            

            # find the mean and variance
            not_polluted_mean = sum(not_polluted_x)/len(not_polluted_x)
            not_polluted_std = statistics.stdev(not_polluted_x)
            for i in range(int(x_part_length*noise_level)):
                # randomly generate a sample
                epsilon = np.random.normal(not_polluted_mean, not_polluted_std, 1)
                epsilon = epsilon[0]
                not_polluted_x.insert(randint(0,len(not_polluted_x)),epsilon)
            noise_x.append(not_polluted_x)
        noise_x = np.array(noise_x)
        noise_x = torch.from_numpy(noise_x.astype(np.float32))
        y_pred = model(noise_x)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()

        total_loss += int(loss.tolist())
        optimizer.step()
    loss_list.append(total_loss/n_batches)
    # go to the validation part
    model.eval()

    test_loss = 0
    test_batches = 0

    for batch_test, (X_test, y_test) in enumerate(test_dataloader):
        test_batches += 1
        y_test_pred = model(X_test)
        loss_test = loss_fn(y_test_pred, y_test)
        optimizer.zero_grad()

        test_loss += int(loss_test.tolist())
    test_losses_list.append(test_loss/test_batches)

    if (epoch%10 == 0):
        print('The number of epoch is:')
        print(epoch)
        print('The current train loss is:')
        print(total_loss/n_batches)
        print('The current test loss is:')
        print(test_loss/test_batches)



plt.plot(np.arange(max_iter),np.array(loss_list),label='train')
plt.plot(np.arange(max_iter),np.array(test_losses_list),label='test')
plt.legend(loc='best')
plt.show()




# predict:
def lastest_data(tinker):
    # target date range
    period1 = int(time.mktime(datetime.datetime(2022, 11, 22, 23, 27).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2023, 4, 11, 23, 27).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{tinker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    L = list(df["Close"])
    res = []
    for s in L:
        res.append(float(s))
    return res #return a list


lastest_price = lastest_data(ticker)

lastest_price = [x for x in lastest_price if str(x) != 'nan']


lastest_price = differencing(lastest_price)
valid_latest = np.array(list(lastest_price[-total_train_size:]))
last_day = lastest_price[-1]
input_torch = valid_latest.astype(np.float32)
input_torch = torch.from_numpy(input_torch)
input_torch = input_torch.unsqueeze(0)

latest_prediction = model(input_torch)


print(latest_prediction[0].tolist())
print(statistics.mean(latest_prediction[0].tolist()))
if (len(latest_prediction[0].tolist()) > 1):
    print(statistics.stdev(latest_prediction[0].tolist()))
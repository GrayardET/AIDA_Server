from aida.aida import *;
import sys;
import pandas as pd;
import numpy as np;
import torch;
import time;
import torch.nn as nn;
import collections;
from aidas.models import biLSTM;
host = 'localhost'; dbname = 'sf01'; user = 'sf01'; passwd = 'sf01'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);

st = time.time()
dw.scaler_dict = {}
dw.X_train_data = {}
dw.X_test_data = {}
dw.y_train_data = {}
dw.y_test_data = {}
new_goog_df= dw.goog.project(('date', 'open', 'high', 'low', 'close'))
new_meta_df= dw.meta.project(('date', 'open', 'high', 'low', 'close'))
new_aapl_df= dw.aapl.project(('date', 'open', 'high', 'low', 'close'))
new_nflx_df= dw.nflx.project(('date', 'open', 'high', 'low', 'close'))
new_amzn_df= dw.amzn.project(('date', 'open', 'high', 'low', 'close'))

def create_moving_averages_columns(tblrData):
    import pandas as pd
    import copy
    days_for_moving_averages = [10, 50, 100]
    data = copy.copy(tblrData.rows);
    # convert ndarray to pandas.dataFrame
    close_df = pd.DataFrame(data['close'],columns=['close'])
    # calculate moving average stat
    for moving_averages in days_for_moving_averages:
        mv_average_col = data[f'MA for {moving_averages} days'] = np.empty(tblrData.numRows, dtype=float)
        average = close_df['close'].rolling(moving_averages).mean()
        for i in range(0, tblrData.numRows):
            mv_average_col[i] = average[i]
    # calculate pct stat
    pct_col = data[f'Daily Return'] = np.empty(tblrData.numRows, dtype=float)
    pct = close_df['close'].pct_change()
    for i in range(0, tblrData.numRows):
            pct_col[i] = pct[i]
    return data

new_goog_df = new_goog_df._U(create_moving_averages_columns);
new_meta_df = new_meta_df._U(create_moving_averages_columns);
new_aapl_df = new_aapl_df._U(create_moving_averages_columns);
new_nflx_df = new_nflx_df._U(create_moving_averages_columns);
new_amzn_df = new_amzn_df._U(create_moving_averages_columns);
print(new_meta_df.tail())

def scaling_close(tblrData,dw,name):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    data = copy.copy(tblrData.rows);
    # convert ndarray to pandas.dataFrame
    close_df = pd.DataFrame(data['close'],columns=['close'])
    scaler = MinMaxScaler(feature_range = (0, 1))
    dw.scaler_dict[name] = copy.copy(scaler)
    close_data = scaler.fit_transform(close_df.values)
    return close_data 

def scale(self,tblrData,company,pred_days):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    prediction_days = pred_days
    data = copy.copy(tblrData.rows);
    # convert ndarray to pandas.dataFrame
    close_df = pd.DataFrame(data['close'],columns=['close'])
    scaler = MinMaxScaler(feature_range = (0, 1))
    self.scaler_dict[company] = copy.copy(scaler)
    close_data = scaler.fit_transform(close_df.values)

    train_size = int(np.ceil(len(close_data) * 0.95))
    print(f'The training size for {company.title()} is {train_size} rows')
    train_data = close_data[0: int(train_size), :]
    test_data = close_data[train_size - prediction_days:, :]

    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(prediction_days, len(train_data)):
        X_train.append(train_data[i - prediction_days: i, 0])
        y_train.append(train_data[i, 0])

    for i in range(prediction_days, len(test_data)):
        X_test.append(test_data[i - prediction_days: i, 0])
        y_test.append(test_data[i, 0])

    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    self.X_train_data[company] = X_train
    self.X_test_data[company] = X_test
    self.y_train_data[company] = y_train
    self.y_test_data[company] = y_test

    return train_size;

dw._X(scale,new_meta_df,'META',30)
dw._X(scale,new_aapl_df,'AAPL',30)
dw._X(scale,new_amzn_df,'AMZN',30)
dw._X(scale,new_nflx_df,'NFLX',30)
dw._X(scale,new_goog_df,'GOOG',30)
#new_meta_df = dw.company_data['META']

#print(type(dw.X_train_data['META']))
#print(size)

en = time.time()
print(en-st)

name = sys.argv[1]
dw.epoch_total = int(sys.argv[2])
dw.epoch_done = 0
model = biLSTM(1, 128, 3)
dw.model = model
dw.X_train_meta_reshaped = torch.tensor(dw.X_train_data['META'], dtype = torch.float).permute(0,2,1)
dw.y_train_meta_reshaped = torch.tensor(dw.y_train_data['META'], dtype = torch.float)
dw.criterion = nn.HuberLoss()
dw.loss = []
dw.time = []

def trainingLoop(dw,iter_num,time_limit,using_GPU):
    start = time.time()
    model = dw.model
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params = parameters)
    criterion = dw.criterion
    optimizer.zero_grad()
    num_finish = 0
    x = dw.X_train_meta_reshaped
    y = dw.y_train_meta_reshaped

    for i in range(iter_num):
        if(time.time() - start < time_limit):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            lossArray = dw.loss
            lossArray.append(loss.to('cpu').detach().numpy())

            dw.lossArray = lossArray
            pred = torch.argmax(y_pred, 1)
            loss.backward()
            optimizer.step()
        else:
            num_finish = i;
            break;
        if(dw.stop):
            num_finish = i;
            logging.info("stopped")
            break;

    dw.model = model
    if( num_finish == 0):
         num_finish = iter_num;
    epoch_done = dw.epoch_done + num_finish
    dw.epoch_done = epoch_done
    end = time.time()
    dw.time.append(end-start)
    return [(time.time() - start),num_finish,psutil.cpu_percent()]

def Condition(dw):
    if dw.epoch_done >= dw.epoch_total:
        return True
    else:
        return False

def Validate(dw,using_GPU):
    pass

result = dw._job(trainingLoop, Condition, Validate, name)
print(dw.time)

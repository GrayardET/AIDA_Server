#same content with Switch_between.py,but in this version the number of iteration is passed to _job(),
#function iterate(): defines what to do in one iteration, so that we can count how many iteration is done in 1s

from aida.aida import *;
import sys;
import pandas as pd;
import numpy as np;
import time as time;
from ctypes import *
import os
import torch;
import torch.nn as nn;
import collections;
host = 'localhost'; dbname = 'bixi'; user = 'bixi'; passwd = 'bixi'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);

name = sys.argv[1]
n = int(sys.argv[2])
dw.loss_target = float(sys.argv[3])
df = pd.DataFrame(np.random.randn(n))
df.columns = ['A']
df['B'] = np.random.randn(n)
df['C'] = np.random.randn(n)
df['D'] = np.random.randn(n)
df['E'] = np.random.randn(n)
df['Y'] = 5 + 3 * df.A + 6 * df.B ** 2 + 7 * df.C ** 3 + 2 * df.D ** 2 + 8 * df.E * df.D + np.random.randn(n)

dataset = df.copy()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Y")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Y')
test_labels = test_dataset.pop('Y')

#train_target = torch.tensor(train_labels.values.astype(np.float32))
#train_target = train_target.view(train_target.shape[0], 1)
#test_target = torch.tensor(test_labels.values.astype(np.float32))
#test_target = test_target.view(test_target.shape[0], 1)

#become a col of np.float32
train_target =train_labels.values.astype(np.float32)
train_target = train_target.reshape(train_target.shape[0], 1)
test_target = test_labels.values.astype(np.float32)
test_target = test_target.reshape(test_target.shape[0], 1)



def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#normed_train_data = torch.from_numpy(normed_train_data.values)
#normed_train_data = normed_train_data.float()
#normed_test_data = torch.from_numpy(normed_test_data.values)
#normed_test_data = normed_test_data.float()

normed_train_data =normed_train_data.values.astype(np.float32)
normed_test_data = normed_test_data.values.astype(np.float32)

def get_training_model(inFeatures=len(train_dataset.keys()), hiddenDim=16, nbClasses=1):
        # construct a shallow, sequential neural network
    model = nn.Sequential(collections.OrderedDict([
        ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
        ("activation_1", nn.ReLU()),
        ("hidden_layer_2", nn.Linear(hiddenDim, hiddenDim)),
        ("activation_2", nn.ReLU()),
        ("output_layer", nn.Linear(hiddenDim, nbClasses))
    ]))
        # return the sequential model
    return model

#def initiate_model:
    #get the model and optimizer
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    #criterion = nn.MSELoss()
    #model = get_training_model()
#return model,optimizer,criterion,epoch_size,epoch_batch,epoch_done

dw.normed_train_data = normed_train_data
dw.train_target = train_target
dw.normed_test_data = normed_test_data
dw.test_target = test_target
dw.epoch_done = 0
dw.criterion = nn.MSELoss()
model = get_training_model()
dw.model = model



def iterate(dw,iter_num,time_limit,using_GPU):
    model = dw.model
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    normed_train_data = dw.normed_train_data
    train_target = dw.train_target
    criterion = dw.criterion
    start = time.time()
    num_finish = 0;

    for i in range (iter_num):
        if(time.time() - start < time_limit):
            predicted = model(normed_train_data)
            loss = criterion(predicted, train_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            num_finish = i;
            break;
        if(dw.stop):
            num_finish = i;
            break;
        #if(using_GPU and dw.stop):
        #    num_finish = i;
        #    logging.info("gpu: stop at iter:"+str(i))
        #    break;

        #used to stop for a short while
        #while(dw.stop):
        #    time.sleep(0.4)
        #    logging.info("cpu: stop at iter:"+str(i))
        #    logging.info("cpu: stop at time"+str(time.time()-start))

    dw.model = model
    if( num_finish == 0):
         num_finish = iter_num;
    epoch_done = dw.epoch_done + num_finish
    dw.epoch_done = epoch_done
    logging.info("total iter:"+str(num_finish))
    logging.info("totally end time"+ str(time.time() - start))
    return [(time.time() - start),num_finish,psutil.cpu_percent()]

def condition(dw,using_GPU):
    normed_test_data = dw.normed_test_data
    test_target = dw.test_target
    predicted = dw.model(normed_test_data)
    loss = dw.criterion(predicted, test_target)
    if loss < dw.loss_target:
        return True
    else:
        logging.info("loss" + str(loss))
        return False


def test_model(dw,using_GPU):
    normed_test_data = dw.normed_test_data
    test_target = dw.test_target
    predicted = dw.model(normed_test_data)
    loss = dw.criterion(predicted, test_target)
    #return_mesg = "the loss of the model is: " + str(loss)
    return_mesg = ""
    return return_mesg


#return_mesg = dw._append(iterate,condition,test_model,name)
return_mesg = dw._job(iterate,condition,test_model,name)
print(name+ " time: " + str(return_mesg))

from aida.aida import *;
import sys;
import pandas as pd;
import numpy as np;
import torch;
import torch.nn as nn;
import collections;
host = 'localhost'; dbname = 'sf01'; user = 'sf01'; passwd = 'sf01'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);

name = sys.argv[1]
n = 50000
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
dw.epoch_size = int(sys.argv[2])
dw.epoch_batch = int(sys.argv[3])


def trainingLoop(dw):
    model = dw.model
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    normed_train_data = dw.normed_train_data
    train_target = dw.train_target
    criterion = dw.criterion
    for epoch in range(dw.epoch_batch):
        predicted = model(normed_train_data)
        loss = criterion(predicted, train_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    normed_test_data = dw.normed_test_data
    test_target = dw.test_target
    predicted = model(normed_test_data)
    loss = dw.criterion(predicted, test_target)
    return_mesg = "the loss of the model is: " + str(loss)
    dw.model = model
    epoch_done = dw.epoch_done + dw.epoch_batch
    dw.epoch_done = epoch_done
    return return_mesg

def condition(dw):
    if dw.epoch_done >= dw.epoch_size: 
        return True
    else:
        return False

def test_model(dw):
    normed_test_data = dw.normed_test_data
    test_target = dw.test_target
    predicted = dw.model(normed_test_data)
    loss = dw.criterion(predicted, test_target)
    #return_mesg = "the loss of the model is: " + str(loss)
    return_mesg = ""
    return return_mesg

return_mesg = dw._Schedule(trainingLoop,condition,test_model,name)
f = open("./result.txt", "a")
f.write(str(return_mesg)+"\n")
f.close()
print(name+ " time: " + str(return_mesg))

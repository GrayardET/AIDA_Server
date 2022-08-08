from aida.aida import *;
import time;
import sys;
import pandas as pd;
import numpy as np;
import torch;
import torch.nn as nn;
import collections;
host = 'localhost'; dbname = 'bixi'; user = 'bixi'; passwd = 'bixi'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);

#n = 1000
#df = pd.DataFrame(np.random.randn(n))
#df.columns = ['A']
#df['B'] = np.random.randn(n)
#df['C'] = np.random.randn(n)
#df['D'] = np.random.randn(n)
#df['E'] = np.random.randn(n)
#df['Y'] = 5 + 3 * df.A + 6 * df.B ** 2 + 7 * df.C ** 3 + 2 * df.D ** 2 + 8 * df.E * df.D + np.random.randn(n)

# create dummy data for training
x_values = [i for i in range(2000000)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)



def get_model():
    #class linearRegression(torch.nn.Module):
    #    def __init__(self, inputSize, outputSize):
    #        super(linearRegression, self).__init__()
    #        self.linear = torch.nn.Linear(inputSize, outputSize)

    #    def forward(self, x):
    #        out = self.linear(x)
    #        return out
    model = nn.Sequential(collections.OrderedDict([
        ("layer", nn.Linear(1, 1)),
    ]));
    return model

start = time.time()
dw.x_train = x_train
dw.y_train = y_train
dw.epoch_done = 0
dw.criterion = nn.MSELoss()
model = get_model();
dw.model = model

def regression(dw):
    model = dw.model
    x_train = dw.x_train
    y_train = dw.y_train
    criterion = dw.criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000000001)
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    for epoch in range(10000):

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

    # get output from the model, given the inputs
        outputs = model(inputs)

    # get loss for the predicted output
        loss = criterion(outputs, labels)
    # get gradients w.r.t to parameters
        loss.backward()

    # update parameters
        optimizer.step()

    new_var = Variable(torch.Tensor([[4.0]]))
    pred_y = model(new_var)
    return pred_y

return_mesg = dw._X(regression)
end = time.time()
rt = end - start;
print("server time: " + str(return_mesg))
print("client time: " + str(rt))

import pandas as pd
import torch
import time
import numpy as np
from numpy.random import randn
import torch.nn as nn
from collections import OrderedDict
from threading import Thread

def exect(on_GPU):
    script_start = time.time()
    n = 0
    if(on_GPU):
        n = 1000000
        epoch_size = 2000
    else:
        n = 50000
        epoch_size =10000
    df = pd.DataFrame(randn(n))
    df.columns = ['A']
    df['B'] = randn(n)
    df['C'] = randn(n)
    df['D'] = randn(n)
    df['E'] = randn(n)
    df['Y'] = 5 + 3 * df.A + 6 * df.B ** 2 + 7 * df.C ** 3 + 2 * df.D ** 2 + 8 * df.E * df.D + randn(n)

    dataset = df.copy()



    # In[109]:

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # In[110]:

    train_stats = train_dataset.describe()
    train_stats.pop("Y")
    train_stats = train_stats.transpose()

    # In[111]:

    train_labels = train_dataset.pop('Y')
    test_labels = test_dataset.pop('Y')


    # In[113]:

    train_target = torch.tensor(train_labels.values.astype(np.float32))

    # In[114]:

    train_target = train_target.view(train_target.shape[0], 1)

    # In[115]:

    test_target = torch.tensor(test_labels.values.astype(np.float32))

    # In[116]:

    test_target = test_target.view(test_target.shape[0], 1)

    # In[118]:

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # In[119]:

    normed_train_data = torch.from_numpy(normed_train_data.values)
    normed_train_data = normed_train_data.float()

    # In[120]:

    normed_test_data = torch.from_numpy(normed_test_data.values)
    normed_test_data = normed_test_data.float()

    # In[121]:

    def get_training_model(inFeatures=len(train_dataset.keys()), hiddenDim=16, nbClasses=1):
      # construct a shallow, sequential neural network
        model = nn.Sequential(OrderedDict([
            ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
            ("activation_1", nn.ReLU()),
            ("hidden_layer_2", nn.Linear(hiddenDim, hiddenDim)),
            ("activation_2", nn.ReLU()),
            ("output_layer", nn.Linear(hiddenDim, nbClasses))
        ]))
        # return the sequential model
        return model

    # In[122]:

    model = get_training_model()

    # In[123]:

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    # In[124]:

    criterion = nn.MSELoss()

    # In[125]:

    model(normed_train_data).size()

    # In[126]:
    st = time.time()
    if(on_GPU):
        model = model.to(torch.device("cuda:0"))
        normed_train_data = normed_train_data.to(torch.device("cuda:0"))
        train_target = train_target.to(torch.device("cuda:0"))
    en = time.time()
    start_time = time.time()
    for epoch in range(epoch_size):
        predicted = model(normed_train_data)
        loss = criterion(predicted, train_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    end_time = time.time()
    execution_time = end_time - start_time
    #2000000
    #return_mesg = "The execution time on GPU for a dataset of size 10000 and 50000 epochs using Pytorch is:" + str(execution_time)
    # In[127]:
    if(on_GPU):
        normed_test_data = normed_test_data.to(torch.device("cuda:0"))
        test_target = test_target.to(torch.device("cuda:0"))
    predicted = model(normed_test_data)
    loss = criterion(predicted, test_target)
    #return_mesg = return_mesg + " and the loss of the model is: " + str(loss)
    script_end = time.time()
    return_mesg = "total time"+str(script_end - st )
    print(return_mesg)

def mult():
    st = time.time()
    A = np.random.rand(2000,30)
    B = np.random.rand(30,7)

    for x in range(300000):
        C = A.dot(B)
    en = time.time()
    print("mult:"+str(en-st))


#threadA = Thread(target=exect, args=(False,))
threadB = Thread(target=exect, args=(False,))
#threadB = Thread(target=mult, args=())
#threadA.start()
threadB.start()
#threadA.join()
threadB.join()

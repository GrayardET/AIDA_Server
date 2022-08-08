import pandas as pd
import torch
import time
import numpy as np
from numpy.random import randn
import torch.nn as nn
from collections import OrderedDict
from threading import Thread
import sys
from pykeops.torch import LazyTensor

def NN(on_GPU):
    script_start = time.time()
    n = 0
    if(on_GPU):
        n = 1000000
        epoch_size = 2000
    else:
        n = 50000
        epoch_size = 10000
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
    return_mesg = "ktotal time"+str(script_end - st )
    print(return_mesg)

def mult():
    st = time.time()
    A = np.random.rand(2000,30)
    B = np.random.rand(30,7)

    for x in range(300000):
        C = A.dot(B)
    en = time.time()
    print("mult:"+str(en-st))

def kmeans():
    np.random.seed(123)
    N,D,K,Niter = 1000000, 2, 50, 10000
    dtype = torch.float32 
    finishedEpoch = 0
    c, cl = 0, 0
    start = time.time()
    device_id = "cuda:0"
    x = (0.7 * torch.randn(N, D, dtype=dtype, device='cuda:0') + 0.3).to(device_id)
    k_means(x, c, cl, K, Niter=Niter, if_gpu = True)
    end = time.time()
    print("kmeans"+str(end-start))

def k_means(x, c=0, cl=0,  K=10, Niter=10, verbose=False, if_gpu=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    if isinstance(c, int) and c == 0:
        c = x[:K, :].clone()  # Simplistic initialization for the centroids

    if if_gpu:
        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
    else:
        x_i = x.view(N, 1, D)  # (N, 1, D) samples
        c_j = c.view(1, K, D)  # (1, K, D) centroids

        # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c

threadA = Thread(target=kmeans, args=())
#threadA = Thread(target=NN, args=(False,))
#threadB = Thread(target=NN, args=(False,))
#threadB = Thread(target=mult, args=())
threadA.start()
#threadB.start()
threadA.join()
#threadB.join()

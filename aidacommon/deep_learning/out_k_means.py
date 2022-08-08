from aida.aida import *;
import sys
import numpy as np
import os.path
import pandas
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor


host = 'yz_Server2'
dbname='sf01'
user='sf01'
passwd='sf01'
jobName='xgboost'
port=55660

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)
name = sys.argv[1]
Niter = int(sys.argv[2])

st_data = time.time()
new_lineitem_df= dw.lineitem.project(('l_quantity', 'l_extendedprice')).filter(Q('l_quantity',10,CMP.LT))
en_1 = time.time()
print("read to aida:"+str(en_1-st_data))
def process(self,data):
    st_1 = time.time()
    data = copy.copy(data.rows);
    st_2 = time.time()
    logging.info("copy data"+str(st_2-st_1))
    data_df = pd.DataFrame(data,columns=['l_quantity', 'l_extendedprice'])
    norm_df = (data_df-data_df.mean())/data_df.std()
    x = torch.tensor(norm_df.values.astype(np.float32))
    x = x.view(x.shape[0],2)
    en_2 = time.time()
    logging.info("process data"+str(en_2-st_2))
    self.x = x.to("cuda:0")
    #self.x = x
dw._X(process,new_lineitem_df)
en_data = time.time()
print("data transfer" + str(en_data-st_data))
dw.K, dw.epoch_total = 50, Niter
dw.dtype = torch.float32
dw.timeArray = []
dw.epoch_done = 0

dw.c, dw.cl = 0, 0
dw.finishedEpochArray = []
dw.LazyTensor = LazyTensor

def k_means(dw, x, c=0, cl=0,  K=10, Niter=10, verbose=False, if_gpu=True):
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    num_finish = -1

    if isinstance(c, int) and c == 0:
        c = x[:K, :].clone()  # Simplistic initialization for the centroids
    elif c.size()[1] != 2:
        c = torch.transpose(c,0,1)
    if if_gpu:
        x_i = dw.LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = dw.LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
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
            #logging.info("cl"+str(cl))

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K,1)
        c /= Ncl  # in-place division to compute the average

    return cl, c

dw.k_means = k_means

def trainingLoop(dw):
    start = time.time()
    cl = dw.cl
    c = dw.c
    k_means = dw.k_means
    K, dtype = dw.K, dw.dtype
    device_id = "cuda:0"
    #remaining = dw.Niter - dw.finishedEpoch
    x = dw.x
    Niter = dw.epoch_total
    cl, c = k_means(dw,x, c, cl, K, Niter=Niter, if_gpu = True)
    dw.cl = cl
    dw.c = c
    end = time.time()
    dw.timeArray.append(end-start)
    dw.finishedEpochArray.append(dw.epoch_total)
    return end - start 


timeUsed = dw._X(trainingLoop)
print(dw.timeArray)
print(dw.finishedEpochArray)
Niter = dw.epoch_total
print("Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(Niter, timeUsed, Niter, timeUsed / Niter))

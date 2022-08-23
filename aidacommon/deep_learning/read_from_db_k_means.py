from aida.aida import *;
import GPUtil
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
#new_lineitem_df= dw.lineitem.project(('l_quantity', 'l_extendedprice')).head(head_num)
en_1 = time.time()
print("read to aida:"+str(en_1-st_data))
def process(self,data):
    data = copy.copy(data.rows);
    st_2 = time.time()
    #logging.info("copy data"+str(st_2-st_1))
    data_df = pd.DataFrame(data,columns=['l_quantity', 'l_extendedprice'])
    data_df = data_df.head(100000)
    norm_df = (data_df-data_df.mean())/data_df.std()
    x = torch.tensor(norm_df.values.astype(np.float32))
    x = x.view(x.shape[0],2)
    if x.size()[1] != 2:
        x = torch.transpose(x,0,1)
    en_2 = time.time()
    logging.info("process data"+str(en_2-st_2))
    self.x = x
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

def k_means(dw, time_limit, x, c=0, cl=0,  K=10, Niter=10, verbose=False, if_gpu=True):
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    num_finish = -1

    if isinstance(c, int) and c == 0:
        c = x[:K, :].clone()  # Simplistic initialization for the centroids
        logging.info(' x size'+str(x.size()))
        logging.info('c size'+str(c.size()))
        logging.info('K is'+str(K))
    elif c.size()[1] != 2:
        c = torch.transpose(c,0,1)
    if if_gpu:
        x_i = dw.LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        logging.info('size'+str(c.size()))
        c_j = dw.LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
    else:
        x_i = x.view(N, 1, D)  # (N, 1, D) samples
        c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        
        if(time.time() - start < time_limit):
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
        else:
            num_finish = i;
            break;
        if(dw.stop):
            num_finish = i;
            break;
    if(num_finish == -1):
        num_finish = Niter;

    return cl, c,num_finish

dw.k_means = k_means
dw.stop = False

def trainingLoop(dw,iter_num,time_limit,using_GPU):
    start = time.time()
    cl = dw.cl
    c = dw.c
    print(c)
    k_means = dw.k_means
    K, dtype = dw.K, dw.dtype
    #device_id = "cuda:0"
    #remaining = dw.Niter - dw.finishedEpoch
    x = dw.x
    Niter = iter_num
    cl, c,num_finish = k_means(dw,time_limit,x, c, cl, K, Niter=Niter, if_gpu = using_GPU)
    dw.cl = cl
    dw.c = c
    end = time.time()
    dw.timeArray.append(end-start)
    dw.epoch_done += num_finish
    dw.finishedEpochArray.append(dw.epoch_done)
    gpus = GPUtil.getGPUs()
    return [(time.time() - start),num_finish,gpus[1].load*100,psutil.cpu_percent()]

def Condition(dw,use_GPU):
    if dw.epoch_done >= dw.epoch_total:
        return True
    else:
        return False

def Testing(dw,using_GPU):
    pass

timeUsed = dw._X(trainingLoop,dw.epoch_total,1000000000,False)
#timeUsed = dw._job(trainingLoop, Condition, Testing, name)
print(dw.timeArray)
print(dw.finishedEpochArray)
Niter = dw.epoch_done
print("Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(Niter, timeUsed, Niter, timeUsed / Niter))

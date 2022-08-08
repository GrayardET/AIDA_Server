import psutil
import time
import GPUtil

#for i in range(10):
#    print(float(psutil.cpu_percent()))
#    time.sleep(0.5)
#for i in range(4):
#    p = psutil.Process(10824)
#    print(p.cpu_percent(interval=0.5))
#    print("total:"+str(float(psutil.cpu_percent())))
#    time.sleep(0.5)


#print(psutil.cpu_count())
gpus = GPUtil.getGPUs()
print("id:" + str(gpus[1].id)+"util:"+ str(gpus[1].load))

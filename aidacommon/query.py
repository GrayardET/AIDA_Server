from multiprocessing import Pool 
import logging
import time
import psycopg2
logging.basicConfig(level=logging.INFO, filename='query.log')
import numpy as np
import psutil

connection=psycopg2.connect(user='sf01',password='sf01',host='localhost',database='sf01')
cursor=connection.cursor()
print(cursor.execute("SELECT pg_backend_pid();"))
number = 50
lengthArr = []
cpuArr = []
index = 1
t_start = time.time()
print(str(float(t_start)))
while True:
    cpu = float(psutil.cpu_percent())
    t1=time.time()
    #cpu: 4%
    #cursor.execute("SELECT * FROM part LIMIT 200;")
    #cpu: 4%
    #cursor.execute("SELECT * FROM part LIMIT 1000;")
    #cpu: 20%
    #cursor.execute("SELECT * FROM part;")
    #cpu: 35%
    cursor.execute("SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part WHERE p_partkey = l_partkey AND p_brand = '[BRAND]' AND p_container = '[CONTAINER]' AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey);")
    #cursor.execute("SELECT ps_partkey,SUM(ps_supplycost * ps_availqty) AS value FROM partsupp,supplier,nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY' GROUP BY ps_partkey HAVING SUM(ps_supplycost * ps_availqty) > (SELECT SUM(ps_supplycost * ps_availqty) * 0.0001000000e-2 FROM partsupp,supplier,nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY') ORDER BY value DESC LIMIT 10")
    length = float(time.time() - t1)
    lengthArr.append(length)
    cpuArr.append(cpu)
    index  += 1
    if index == number:
        index = 0

        # logging.info("start:{}:elapsed:{}".format(t1,np.mean(lengthArr)))
        with open('result_gpu.csv', 'a') as f:
            f.write(str(float(time.time()-t_start)) +','+ str(np.mean(lengthArr)) +','+ str(np.mean(cpuArr))+'\n')
        lengthArr = []
        cpuArr = []


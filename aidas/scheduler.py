from abc import ABCMeta;
import signal;
from collections import deque;
import threading;
import GPUtil;
import psutil;
import time;
import bisect;
from timeloop import Timeloop;
from datetime import timedelta;
from tblib import pickling_support;
pickling_support.install();
import dill as custompickle;

import logging;

from aidacommon.aidaConfig import AConfig;

class ScheduleManager(metaclass=ABCMeta):
    """Singleton class, there will be only one schedule manager in the system"""
    __ClassLock = threading.RLock();
    __ScheduleManagerObj = None;

    @staticmethod
    def getScheduleManager():
        class __ScheduleManager:
            """Class that dispatching jobs"""

            __RepoLock = threading.RLock();
            __maybe_available_Lock = threading.Lock();
            __maybe_available = threading.Condition(__maybe_available_Lock);
            __GPUQueue = [];       # [DBC,cv] waiting queue for jobs can run both on CPU and GPU
            __CPUQueue = [];       # [DBC,cv] waiting queue for jobs can only run on CPU
            __GPU_long_Queue = []; # waiting queue without explicit epoch num
            __CPU_long_Queue = []; # waiting queue without explicit epoch num
            __GPU_inuse = deque();      # a cv queue: the job using the GPU, for
            __CPU_inuse = deque();      # a cv queue: the task using the CPU
                                        # these 2 queues are used when checking if the job/task
                                        # should go to CPU or GPU 
            __GPU_paused = deque();
            __CPU_paused = deque();
            __CPU_inuse_name = deque(); # a string queue: since different task from same job use 
                                        # different cv, need a queue to record the job name
                                        # to insert next task to the head of GPUQ in 
                                        # Dictator Strategy
            __gpu_free = True;          # a token to indicate if anyone is using GPU
            __cpu_free = True;          # a token to indicate if anyone is using CPU
                                        # cannot use len(self.__CPU_inuse) because there might be 
                                        # concurrent issue that two jobs 
                                        # adding to Queue at the same time
            __GPU_test_job = 0;         # recording how many jobs wanting to do the 1s test on gpu
            __CPU_test_job = 0;         # recording how many jobs wanting to do the 1s test on cpu
            __GPU_util = 0;             # gpu_util if running on GPU
            __CPU_util = 0;             # cpu_util if running on CPU and GPU
            __CPU_util_no_CPU_job = 0;  # cpu_util if jobs are only running on GPU
            __GPU_running_job_length = -1;# record the length of current running job
            __CPU_running_job_length = -1;
            __default_exec_time = 10
            __job_limit = 1
            __GPU_job_limit = 1;        # after every 1 normal job,pump one job from __GPU_long_Queue
            __CPU_job_limit = 1;        # after every 1 normal job,pump one job from __CPU_long_Queue

            def __init__(self):
                SchMgrObj = self;


                # the scheduler only wakes up if there is any resource available,
                # or there is new job coming
                def activate_by_job(self):
                    logging.info("add to head scheduler is running");

                    while(True):
                        while(len(self.__GPU_paused) == 0 and len(self.__CPU_paused) == 0 and (len(self.__GPUQueue)> 0 or len(self.__CPUQueue)> 0 or  len(self.__GPU_long_Queue)> 0 or  len(self.__CPU_long_Queue)> 0)):
                            occupied = True
                            deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.9, attempts=1)
                            logging.info("begin schedule")
                            logging.info(str( len(self.__GPUQueue)))
                            logging.info(str(deviceID[0]))
                            logging.info(self.__gpu_free)
                            #incase we have assigned the gpu but the work is not start running
                            if 1 == deviceID[0] and self.__gpu_free:
                                if( len(self.__GPUQueue)> 0 and self.__GPU_job_limit > 0):
                                    self.__gpu_free = False;
                                    triple = self.__GPUQueue.pop(0);
                                    logging.info("poped from gpu queue");
                                    self.__GPU_running_job_length = triple[0]
                                    self.invoke_GPU(triple[1],triple[2]);
                                    occupied = False;
                                    self.__GPU_job_limit  = self.__GPU_job_limit - 1;
                                elif(len(self.__GPU_long_Queue)> 0):
                                    self.__gpu_free = False;
                                    triple = self.__GPU_long_Queue.pop(0);
                                    logging.info("poped from gpu long");
                                    self.__GPU_running_job_length = self.__default_exec_time
                                    self.invoke_GPU(triple[0],triple[1]);
                                    occupied = False;
                                    self.__GPU_job_limit  = self.__job_limit
                            if(self.__cpu_free):
                                logging.info("arrive cpu scheduling")
                                if(len(self.__CPUQueue)> 0 and (len(self.__CPU_long_Queue) == 0 or self.__CPU_job_limit > 0)):
                                    triple = self.__CPUQueue.pop(0);
                                    logging.info("poped from cpu queue")
                                    self.__CPU_running_job_length = triple[0]
                                    self.__CPU_util_no_CPU_job = psutil.cpu_percent()
                                    self.__CPU_util =triple[3]+self.__CPU_util_no_CPU_job
                                    logging.info("cpu job uses:" + str(self.__CPU_util))
                                    self.invoke_CPU(triple[1],triple[2]);
                                    self.__cpu_free = False;
                                    occupied = False;
                                    time.sleep(1)
                                    self.__CPU_job_limit -= 1
                                elif(len(self.__CPU_long_Queue)> 0):
                                    triple = self.__CPU_long_Queue.pop(0);
                                    logging.info("poped from cpu long")
                                    self.__CPU_running_job_length = self.__default_exec_time
                                    self.__CPU_util_no_CPU_job = psutil.cpu_percent()
                                    self.__CPU_util =triple[2]+self.__CPU_util_no_CPU_job
                                    logging.info("cpu job uses:" + str(self.__CPU_util))
                                    self.invoke_CPU(triple[0],triple[1]);
                                    self.__cpu_free = False;
                                    occupied = False;
                                    time.sleep(1)
                                    self.__CPU_job_limit = self.__job_limit
                            if(occupied):
                                    break;
                            logging.info("short snap")
                            time.sleep(1)
                        time.sleep(1)


                def check_q_length(self):
                    pause_cpu = False
                    while(True):
                        cpu_now = psutil.cpu_percent()
                        #logging.info("queue length:"+ str(len(self.__GPUQueue)));
                        logging.info("cpu util:" + str(cpu_now))
                        if(self.__CPU_test_job == 0 and self.__CPU_util >10 and not pause_cpu and cpu_now - self.__CPU_util > 20):
                            logging.info("query comes")
                            for dw in self.__CPU_inuse:
                                dw.stop = True;
                            pause_cpu = True
                        elif(self.__CPU_test_job == 0 and pause_cpu and cpu_now < self.__CPU_util_no_CPU_job + 5):
                            for dw in self.__CPU_paused:
                                dw.stop = False;

                        time.sleep(1)


                #Handle signals to exit gracefully.
                if(threading.current_thread() == threading.main_thread()):
                    signal.signal(signal.SIGINT, self.terminate);
                    signal.signal(signal.SIGTERM, self.terminate);

                #Start the server polling as a daemon thread.
                self.__srvrThread = threading.Thread(target=activate_by_job,args=(self,));
                self.__srvrThread.daemon = True;
                self.__srvrThread.start();
                self.__checkLengthThread = threading.Thread(target=check_q_length,args=(self,));
                self.__checkLengthThread.daemon = True;
                self.__checkLengthThread.start();

            #try to wake up the scheduler if it's asleep    
            def wake_up(self):
                succ_acquire = self.__maybe_available.acquire(False);
                if succ_acquire:
                    self.__maybe_available.notify();
                    self.__maybe_available.release();

            def get_GPU_running_job_length(self):
                return self.__GPU_running_job_length


            def get_CPU_running_job_length(self):
                return self.__CPU_running_job_length


            def get_GPU_waiting(self,exec_time):
                waiting_time = 0;
                #sum up all the waiting time ahead of me
                for tuples in self.__GPUQueue:
                    if(tuples[0] < exec_time):
                        waiting_time +=tuples[0]
                    else:
                        break
                waiting_time += exec_time
                return waiting_time

            def get_CPU_waiting(self,exec_time):
                waiting_time = 0;
                #sum up all the waiting time ahead of me
                for tuples in self.__CPUQueue:
                    if(tuples[0] < exec_time):
                        waiting_time +=tuples[0]
                    else:
                        break
                waiting_time += exec_time
                return waiting_time

            # called when a job is interrupted, release control of gpu
            def finish_GPU_pause(self,dw,name):
                self.__GPU_inuse.remove(dw);
                self.__GPU_paused.append(dw);
                self.__gpu_free = True;
                logging.info("now Gpu queue" + str(len(self.__GPU_inuse)))
                logging.info(name+" finish gpu");
            
            #called when a job is reput to the queue after paused
            def put_back_GPU_to_queue(self,dw):
                self.__GPU_paused.remove(dw);

            # called when a job is totally finished
            def finish_GPU_all(self,dw):
                if(self.__GPU_inuse.count(dw) > 0):
                    self.__GPU_inuse.remove(dw);
                if(self.__GPU_paused.count(dw) > 0):
                    self.__GPU_paused.remove(dw);
                self.__gpu_free = True;

            def finish_CPU_pause(self,dw,name):
                self.__CPU_inuse.remove(dw);
                self.__CPU_paused.append(dw);
                self.__cpu_free = True;
                logging.info(name+" finish cpu");

            def put_back_CPU_to_queue(self,dw):
                self.__CPU_paused.remove(dw);

            def finish_CPU_all(self,dw):
                if(self.__CPU_inuse.count(dw) > 0):
                    self.__CPU_inuse.remove(dw);
                if(self.__CPU_paused.count(dw) > 0):
                    self.__CPU_paused.remove(dw);
                self.__cpu_free = True;

            def insert_GPU(self,condition,name,dw,estimate_time,gpu_util):
                with __ScheduleManager.__RepoLock:
                    bisect.insort(self.__GPUQueue,(estimate_time,condition,dw,gpu_util));
                    logging.info(name+"put to gpu queue");
                    self.wake_up();

            def insert_CPU(self,condition,name,dw,estimate_time,cpu_util):
                with __ScheduleManager.__RepoLock:
                    bisect.insort(self.__CPUQueue,(estimate_time,condition,dw,cpu_util));
                    logging.info(name+"put to cpu queue");
                    logging.info("cpu q length"+ str(len(self.__CPUQueue)))
                    self.wake_up();

            def insert_GPU_long(self,condition,name,dw,gpu_util):
                with __ScheduleManager.__RepoLock:
                    self.__GPU_long_Queue.append((condition,dw,gpu_util));
                    logging.info(name+"put to gpu long ");
                    self.wake_up();

            def insert_CPU_long(self,condition,name,dw,cpu_util):
                with __ScheduleManager.__RepoLock:
                    self.__CPU_long_Queue.append((condition,dw,cpu_util));
                    logging.info(name+"put to cpu long");
                    self.wake_up();


            def append_CPU(self,condition,name,dw):
                with __ScheduleManager.__RepoLock:
                    self.__CPUQueue.append((0,condition,dw));
                    logging.info(name+"put to cpu queue");
                    self.wake_up();

            def append_GPU(self,condition,name,dw):
                with __ScheduleManager.__RepoLock:
                    self.__GPUQueue.append((0,condition,dw));
                    logging.info(name+"put to gpu queue");
                    self.wake_up();


            def in_GPU(self,dw):
                if(self.__GPU_inuse.count(dw) > 0):
                    return True;
                else: return False;

            def invoke_GPU(self,cv,dw):
                self.__GPU_inuse.append(dw);
                while(not cv.acquire()):
                    pass;
                cv.notify();
                cv.release();
                logging.info("send job to GPU");


            def invoke_CPU(self,cv,dw):
                #tasks in GPUQueue has higher priority
                self.__CPU_inuse.append(dw);
                while(not cv.acquire()):
                    pass;
                cv.notify();
                cv.release();
                logging.info("send job to CPU");

            def coming_test_gpu(self):
                logging.info("try to test gpu")
                self.__GPU_test_job += 1;
                if(self.__GPU_test_job == 1):
                    logging.info("dw to stop:" + str(len(self.__GPU_inuse)))
                    for dw in self.__GPU_inuse:
                        logging.info("pause dw:" + str(dw))
                        dw.stop = True;

            def finish_test_gpu(self):
                self.__GPU_test_job -= 1;
                if(self.__GPU_test_job == 0):
                    logging.info("resume current gpu job")
                    for dw in self.__GPU_paused:
                        dw.stop = False;

            def coming_test_cpu(self):
                self.__CPU_test_job += 1;
                if(self.__CPU_test_job == 1):
                    logging.info("pause current cpu job")
                    logging.info(str(len(self.__CPU_inuse)))
                    for dw in self.__CPU_inuse:
                        logging.info("pause dw:" + str(dw))
                        dw.stop = True;

            def finish_test_cpu(self):
                self.__CPU_test_job -= 1;
                if(self.__CPU_test_job == 0):
                    for dw in self.__CPU_paused:
                        dw.stop = False;


            def close(self):
                self.__srvr.shutdown();
                self.__srvr.server_close();

            def terminate(self, signum, frame):
                self.close();

        with ScheduleManager.__ClassLock:
            if (ScheduleManager.__ScheduleManagerObj is None):  # There is no connection manager object currently.
                schmgr = __ScheduleManager();
                ScheduleManager.__ScheduleManagerObj = schmgr;
            logging.info("end of init");

            # Return the connection manager object.
            return ScheduleManager.__ScheduleManagerObj;


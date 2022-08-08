from abc import ABCMeta;
import signal;
from collections import deque;
import threading;
import GPUtil;
import time;
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
            __GPUQueue = deque();       # waiting queue for jobs can run both on CPU and GPU
            __CPUQueue = deque();       # waiting queue for jobs can only run on CPU
            __GPU_inuse = deque();      # a cv queue: the job using the GPU, for
            __CPU_inuse = deque();      # a cv queue: the task using the CPU
                                        # these 2 queues are used when checking if the job/task
                                        # should go to CPU or GPU 
            __CPU_inuse_name = deque(); # a string queue: since different task from same job use 
                                        # different cv, need a queue to record the job name
                                        # to insert next task to the head of GPUQ in 
                                        # Dictator Strategy
            __gpu_free = True;          # a token to indicate if anyone is using GPU
            __cpu_free = True;          # a token to indicate if anyone is using CPU
                                        # cannot use len(self.__CPU_inuse) because there might be 
                                        # concurrent issue that two jobs 
                                        # adding to Queue at the same time

            def __init__(self):
                SchMgrObj = self;
                # still a stub: error, handler,
                # check regularly in case some jobs taking too long in CPU/GPU
                def interval_check(self):
                    while(True):
                        while(len(self.__GPUQueue)> 0 or len(self.__CPUQueue)> 0):
                            if(len(self.__GPUQueue)> 0):
                                cv = self.__GPUQueue.popleft();
                                deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.5, attempts=1)
                                #incase we have assigned the gpu but the work is not start running
                                if 1 == deviceID[0] and self.__gpu_free:
                                    logging.info("try to invoke gpu");
                                    self.invoke_GPU(cv);
                                elif(len(self.__CPU_inuse) == 0):
                                    self.invoke_CPU(cv);
                                else:
                                    break;
                            elif(len(self.__CPUQueue)> 0):
                                if(len(self.__CPU_inuse) == 0):
                                    self.invoke_CPU;
                        time.sleep(5);


                # the scheduler only wakes up if there is any resource available,
                # or there is new job coming
                def activate_by_job(self):
                    logging.info("add to head scheduler is running");

                    while(True):
                        with self.__maybe_available:
                            while((not self.__gpu_free and not self.__cpu_free) or (len(self.__GPUQueue)== 0 and len(self.__CPUQueue)== 0)):
                                self.__maybe_available.wait();
                        #incase too quick to send another request 
                        #before the former is really start working
                        while(len(self.__GPUQueue)> 0):
                            if(len(self.__GPUQueue)> 0):
                                deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.5, attempts=1)
                                #incase we have assigned the gpu but the work is not start running
                                #logging.info(deviceID)
                                if 1 == deviceID[0] and self.__gpu_free:
                                    logging.info("here1")
                                    self.__gpu_free = False;
                                    cv = self.__GPUQueue.popleft();
                                    self.invoke_GPU(cv);
                                #elif(self.__cpu_free):
                                #    cv = self.__GPUQueue.popleft();
                                #    self.invoke_CPU(cv);
                                #    self.__cpu_free = False;
                                else:
                                    break;
                            elif(len(self.__CPUQueue)> 0):
                                if(self.__cpu_free):
                                    cv = self.__CPUQueue.popleft();
                                    self.invoke_CPU(cv);
                                    self.__cpu_free = False;

                def check_q_length(self):
                    while(True):
                        logging.info("queue length:"+ str(len(self.__GPUQueue)));
                        time.sleep(5)

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
                with self.__maybe_available:
                    self.__maybe_available.notify();
                #succ_acquire = self.__maybe_available.acquire(False);
                #if succ_acquire:
                #    self.__maybe_available.notify();
                #    self.__maybe_available.release();

            # clean up stuff
            def finish_GPU(self,condition_var,name):
                self.__GPU_inuse.remove(condition_var);
                self.__gpu_free = True;
                self.wake_up();
                if(self.__CPU_inuse_name.count(name) > 0):
                    self.__CPU_inuse_name.remove(name);
                logging.info(self.__gpu_free);
                logging.info(name+" finish gpu");

            # finish_CPU is called after one task finishes CPU
            # while cleanup_CPU is called when the whole job is finished
            # totally on CPU, i.e. all tasks are in CPU,
            # none of them goes to GPU
            def finish_CPU(self,condition_var,name):
                self.__CPU_inuse.remove(condition_var);
                if(self.__CPU_inuse_name.count(name) == 0):
                    self.__CPU_inuse_name.append(name);
                self.__cpu_free = True;
                self.wake_up();
                logging.info(name+" finish cpu");
                logging.info(self.__CPU_inuse.count(condition_var));

            def cleanup_CPU(self,name):
                if(self.__CPU_inuse_name.count(name) > 0):
                    self.__CPU_inuse_name.remove(name);
            

            # RR strategy:
            # put a job that can be executed on GPU
            def schedule_GPU(self, condition_var,name):
                with __ScheduleManager.__RepoLock:
                    self.__GPUQueue.append(condition_var);
                    logging.info(name+" to tail");
                    self.wake_up();
                    logging.info("end schedule");

            # RR strategy:
            # put a job that cannot be executed on GPU
            def schedule_CPU(self, condition_var):
                with __ScheduleManager.__RepoLock:
                    self.__CPUQueue.append(condition_var);
                    self.wake_up();


            def in_GPU(self,condition_var):
                if(self.__GPU_inuse.count(condition_var) > 0):
                    return True;
                else: return False;

            def invoke_GPU(self,cv):
                self.__GPU_inuse.append(cv);
                while(not cv.acquire()):
                    pass;
                cv.notify();
                cv.release();
                logging.info("to GPU");


            def invoke_CPU(self,cv):
                #tasks in GPUQueue has higher priority
                self.__CPU_inuse.append(cv);
                while(not cv.acquire()):
                    pass;
                cv.notify();
                cv.release();
                logging.info("to CPU");


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


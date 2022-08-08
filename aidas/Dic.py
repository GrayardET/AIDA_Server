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
            __GPU_occupied = deque();
            __GPUQueue = deque();
            __CPUQueue = deque();
            __GPU_inuse = deque();
            __CPU_inuse = deque();
            __CPU_inuse_name = deque();
            __gpu_free = True;
            __cpu_free = True;

            def __init__(self):
                SchMgrObj = self;
                def interval_check(self):
                    while(True):
                        while(len(self.__GPUQueue)> 0 or len(self.__CPUQueue)> 0):
                            if(len(self.__GPUQueue)> 0):
                                cv = self.__GPUQueue.popleft();
                                deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.5, attempts=1)
                                #incase we have assigned the gpu but the work is not start running
                                if 1 == deviceID[0] and self.__GPU_occupied.count(1) == 0:
                                    logging.info("try to invoke gpu");
                                    self.__GPU_occupied.append(1);
                                    self.invoke_GPU(cv);
                                elif(len(self.__CPU_inuse) == 0):
                                    self.invoke_CPU(cv);
                                else:
                                    break;
                            elif(len(self.__CPUQueue)> 0):
                                if(len(self.__CPU_inuse) == 0):
                                    self.invoke_CPU;
                        time.sleep(5);


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
                                if 1 == deviceID[0] and self.__gpu_free:
                                    self.__gpu_free = False;
                                    cv = self.__GPUQueue.popleft();
                                    self.invoke_GPU(cv);
                                elif(self.__cpu_free):
                                    cv = self.__GPUQueue.popleft();
                                    self.invoke_CPU(cv);
                                    self.__cpu_free = False;
                                else:
                                    break;
                            elif(len(self.__CPUQueue)> 0):
                                if(self.__cpu_free):
                                    cv = self.__CPUQueue.popleft();
                                    self.invoke_CPU(cv);
                                    self.__cpu_free = False;

                #Handle signals to exit gracefully.
                if(threading.current_thread() == threading.main_thread()):
                    signal.signal(signal.SIGINT, self.terminate);
                    signal.signal(signal.SIGTERM, self.terminate);

                #Start the server polling as a daemon thread.
                self.__srvrThread = threading.Thread(target=activate_by_job,args=(self,));
                self.__srvrThread.daemon = True;
                self.__srvrThread.start();
            def wake_up(self):
                with self.__maybe_available:
                    self.__maybe_available.notify();

            def finish_GPU(self,condition_var,name):
                self.__GPU_inuse.remove(condition_var);
                self.__gpu_free = True;
                self.wake_up();
                if(self.__CPU_inuse_name.count(name) > 0):
                    self.__CPU_inuse_name.remove(name);
                logging.info(name+" finish gpu");

            def finish_CPU(self,condition_var,name):
                self.__CPU_inuse.remove(condition_var);
                if(self.__CPU_inuse_name.count(name) == 0):
                    self.__CPU_inuse_name.append(name);
                self.__cpu_free = True;
                #self.wake_up();
                logging.info(name+" finish cpu");
                logging.info(self.__CPU_inuse.count(condition_var));

            def cleanup_CPU(self,name):
                if(self.__CPU_inuse_name.count(name) > 0):
                    self.__CPU_inuse_name.remove(name);
            

            def schedule_GPU(self, condition_var,name):
                with __ScheduleManager.__RepoLock:
                    if(self.__CPU_inuse_name.count(name) > 0):
                        self.__GPUQueue.appendleft(condition_var);
                        logging.info(name+" to head");
                    else:
                        self.__GPUQueue.append(condition_var);
                        logging.info(name+" to tail");
                    self.wake_up();
                    logging.info("end schedule");

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


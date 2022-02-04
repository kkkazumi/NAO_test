#from naoqi import *
import nao_motion

import time 
import datetime
import numpy as np
import threading

import tensorflow as tf

#import Queue

from face_module import get_face
from neural_net import *

import itertools
import fasteners

epsilon = 0.1
mu = 0.9
epoch = 10

TIME_SIG = 4#four-four
BARS = 8

HOST = "127.0.0.1"
PORT= 41849

ANGLE_DIM= 2
FACE_DIM= 1
LEARN_NUM = 2

ACT_GEN_WIDTH = 4

LOOP = TIME_SIG *2
INPUT_DIM = ANGLE_DIM * LOOP
OUTPUT_DIM = FACE_DIM

#time of start learning
ML_TIME = LOOP*2

class Motion:
  _lock = threading.Lock()
  #_lock = fasteners.InterProcessLock('/var/tmp/lockfile')
  def __init__(self,robot_name,robot,robot_move_func,face_file,N):
    self.robot= robot
    self.robot_move_func= robot_move_func
    self.motion_cand = np.zeros(N)

    self.better_motion_file = "better_motion.csv"

    self.neural = Neural(robot_name,INPUT_DIM,OUTPUT_DIM,LEARN_NUM)
    self._neural = Neural(robot_name+"o",INPUT_DIM,OUTPUT_DIM,LEARN_NUM)
    self.neural.get_model("new")
    self._neural.get_model("new")
    self.graph = tf.get_default_graph()

    self.face_history = np.zeros((100,FACE_DIM))
    self.angle_history = np.zeros((100,ANGLE_DIM))

    self.count = 0

    self.face_file = face_file

  def ml_loop(self):
    mark=0
    with self.graph.as_default():
      if(self.count<ML_TIME):
        mode="new"
      else:
        mode="present"
      while(True):
        if(self.count%int(ML_TIME)==0):
          if(mark==0):
            input_array = np.reshape(np.hstack((self.angle_history[-LOOP:,:])),(-1,INPUT_DIM))
            output_array = np.reshape(np.hstack((self.face_history[-1,:])),(-1,OUTPUT_DIM))
            print("before learning",mode,mark)
            #self._neural.train(np.reshape(input_array,(-1,INPUT_DIM)),np.reshape(output_array,(-1,OUTPUT_DIM)),mode,self.count)  
            Motion._lock.acquire()
            self.neural.model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["accuracy"])
            self.neural.model.fit(x=input_array, y=output_array, nb_epoch=2)
            Motion._lock.release()
            mark=1
        else:
          mark = 0

  def update_better_motion(self):
    #candidate_score = np.zeros(split_size**N)
    with self.graph.as_default():
      #candidate_array = self.set_candidate_array(1)
      #self.candidate_array
      reshape=np.reshape(self.candidate_array,[-1,INPUT_DIM])
      candidate_score = self.neural.predict(reshape)
      better_motion = candidate_array[np.argmax(candidate_score),-ANGLE_DIM*TIME_SIG:]

      np.savetxt(better_motion,self.better_motion_file,delimiter=",")

  def get_motion(self):
    candidate_score = self.neural.predict(reshape)
    better_motion = candidate_array[np.argmax(candidate_score),-ANGLE_DIM*TIME_SIG:]
    return better_motion

  def robot_move(self,motion_order):
    for t in range(TIME_SIG):
      self.robot_move_func(self.robot,motion_order[t])
      #self.robot_move_func(self.robot,np.reshape(angle[t,:],(-1,ANGLE_DIM)),t)

      #checking face here
      if(os.path.exists(self.face_file)==True):
        if(os.stat(self.face_file).st_size!=0):
          face = np.loadtxt(self.face_file,delimiter=",")

          self.face_history[:-1,:] = self.face_history[1:,:]
          self.angle_history[:-1,:] = self.angle_history[1:,:]

          self.face_history[-1,:] = face[-1,2]
          self.angle_history[-1,:] = angle[t,:]

      else:
        print("no face")

      #time.sleep(1.0)

  def motion_loop(self):
    self.count = 0
    next_motion = self.get_motion()

    for i in range(BARS):
      self.count = i

      mov_thr = threading.Thread(target=self.robot_move,args=(next_motion,))
      mov_thr.start()

      next_motion = self.get_motion()
      mov_thr.join()

  def run(self):

    th_robot = threading.Thread(target=self.motion_loop,args=())
    th_learning = threading.Thread(target=self.ml_loop,args=())

    th_learning.setDaemon(True)

    th_robot.start()
    th_learning.start()

    th_robot.join()
    #th_learning.join()

def main():
  robot_name,robot,robot_move_func,N=nao_motion.nao_data()
  face_file = "/home/kazumi/prog/emopy_test/test_face.csv"

  motion_main=Motion(robot_name,robot,robot_move_func,face_file,N)
  motion_main.run()

if __name__ == '__main__':
  main()

  #q_face= Queue.Queue()
  #th_face = threading.Thread(target=get_face,args=(q_face,))

  #th_face.setDaemon(True)

from naoqi import *
import time 
import datetime
import numpy as np
import threading

import Queue

from face_module import get_face
from neural_net import *

import itertools

epsilon = 0.1
mu = 0.9
epoch = 10

TIME_SIG = 4#four-four
MEASURES = 4
TRIAL_NUM = TIME_SIG * MEASURES
BPM = 100

HOST = "127.0.0.1"
PORT= 38403

ANGLE_DIM= 2
FACE_DIM= 1
LEARN_NUM = 2

ACT_GEN_WIDTH = 4

LOOP = 2
INPUT_DIM = ANGLE_DIM * LOOP

def conv_angle(key_str,angle):
    key_list = ["HeadYaw","HeadPitch","LShoulderRoll", "RShoulderRoll","LShoulderPitch", "RShoulderPitch","LElbowRoll", "RElbowRoll","LElbowYaw", "RElbowYaw"]
    angle_array = 0.5*np.array([[-2.0,2.0],[-0.6,0.5],[-0.3,1.3],[-1.3,0.3],[-2.0,2.0],[-2.0,2.0],[-1.5,0],[0,1.5],[-2.0,2.0],[-2.0,2.0]])
    index=key_list.index(key_str)
    a = abs(angle_array[index,0])+abs(angle_array[index,1])
    b = angle_array[index,0]
    return angle*a + b


def move(proxy,angle,t):
    speed = t
    #motion_proc.changeAngles(["HeadYaw", "HeadPitch"], [conv_angle("HeadYaw",angle[0,0]), conv_angle("HeadPitch",angle[0,1])], speed)
    proxy.changeAngles(["LShoulderRoll", "RShoulderRoll"], [conv_angle("LShoulderRoll",angle[0,0]), conv_angle("RShoulderRoll",angle[0,1])],speed )
    #motion_proc.changeAngles(["LShoulderPitch", "RShoulderPitch"], [conv_angle("LShoulderPitch",angle[0,0]), conv_angle("RShoulderPitch",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowRoll", "RElbowRoll"], [conv_angle("LElbowRoll",angle[0,0]), conv_angle("RElbowRoll",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowYaw", "RElbowYaw"], [conv_angle("LElbowYaw",angle[0,0]), conv_angle("RElbowYaw",angle[0,1])],speed)

class Motion:
  def __init__(self,robot_name,robot,robot_move_func):
    self.robot= robot
    self.robot_move_func= robot_move_func

    self.neural = Neural(robot_name,ANGLE_DIM*2,FACE_DIM,LEARN_NUM)
    self.neural.get_model("new")

    self.face_history = np.zeros((100,FACE_DIM))
    self.angle_history = np.zeros((100,ANGLE_DIM))

    self.count = 0

  def run(self):

    th_robot = threading.Thread(target=self.motion_loop,args=())
    th_learning = threading.Thread(target=self.ml_loop,args=())

    th_learning.setDaemon(True)

    th_robot.start()
    th_learning.start()

    th_robot.join()
    #th_learning.join()



  def get_motion_para(self,present_angle,i):
    t= np.random.rand()
    t = 60.0/float(BPM)

    if(i==0):
      angle= np.random.rand(ANGLE_DIM,1)
    else:
      x = int(ACT_GEN_WIDTH)
      l=np.linspace(0,1,x)

      candidate_array = np.zeros((x**ANGLE_DIM,ANGLE_DIM*2))
      candidate_score = np.zeros(x**ANGLE_DIM)

      index = 0

      for v in itertools.product(l,repeat=ANGLE_DIM):
        candidate_array[index,:] = np.hstack((present_angle,np.array(v)))
        reshape=np.reshape(candidate_array[index,:].T,[-1,INPUT_DIM])

        candidate_score[index] = self.neural.predict(reshape)
        index+=1

      angle = candidate_array[np.argmax(candidate_score),-ANGLE_DIM:]

    return t,np.reshape(angle,(-1,ANGLE_DIM))

  #def ml_loop(self,present_angle,angle,face):
  def ml_loop(self):
    while(True):
      if(self.count%10==0):
        input_array = self.angle_history[-LOOP:,:]
        print("input_array",input_array,np.hstack((input_array)))
        output_array = self.face_history[-LOOP:,:]
        #self.neural.train(np.reshape(np.hstack((present_angle,angle)),(-1,ANGLE_DIM*2)),np.reshape(face,(-1,FACE_DIM)))  

        #self.neural.train(np.reshape(np.hstack((present_angle,angle)),(-1,ANGLE_DIM*2)),np.reshape(face,(-1,FACE_DIM)))  

  def motion_loop(self):
    present_angle = np.zeros(ANGLE_DIM)
    self.robot_move_func(self.robot,np.reshape(present_angle,(-1,ANGLE_DIM)),1)
    face_file = "/home/kazumi/prog/emopy_test/test_face.csv"

    for i in range(TRIAL_NUM):
      self.count = i
      t,angle = self.get_motion_para(present_angle,0)
      #t,angle = get_angle(present_angle,neural,self.count)

      self.robot_move_func(self.robot,angle,t)

      #face2##
      if(os.path.exists(face_file)==True):
        if(os.stat(face_file).st_size!=0):
          face = np.loadtxt(face_file,delimiter=",")
          #print("face",face[-1,:3],face.shape)
          #if(face.shape>500):
          #  os.remove(face_file)
          #  print("delete")

          self.face_history[:-1,:] = self.face_history[1:,:]
          self.angle_history[:-1,:] = self.angle_history[1:,:]

          self.face_history[-1,:] = face[-1,2]
          self.angle_history[-1,:] = angle

      else:
        print("no face")

        #learning(neural,present_angle,angle[0,:],face)
        #with open("learning.txt",'a') as f:
        #  f.write(str(present_angle)+str(angle[0,:])+","+str(face)+"\n")

      present_angle = np.reshape(angle,(ANGLE_DIM))

      time.sleep(t)

def main():
  
  robot_name = "NAO"
  robot = ALProxy("ALMotion", HOST, PORT)
  robot_move_func = move
  motion_main=Motion(robot_name,robot,robot_move_func)
  motion_main.run()

if __name__ == '__main__':
  main()

  #q_face= Queue.Queue()
  #th_face = threading.Thread(target=get_face,args=(q_face,))

  #th_face.setDaemon(True)

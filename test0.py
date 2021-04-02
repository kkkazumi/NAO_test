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

TRIAL_NUM = 100
HOST = "127.0.0.1"
PORT= 39649

ANGLE_DIM= 2
FACE_DIM= 1
LEARN_NUM = 5

ACT_GEN_WIDTH = 4

LOOP = 5
INPUT_DIM = ANGLE_DIM * LOOP

#def learning(neural,present_angle,angle,face):
def learning(neural,input_angle,face):
  neural.train(np.reshape(input_angle,(-1,INPUT_DIM)),np.reshape(face,(-1,FACE_DIM)))  

def get_angle(angle_history,neural,i):
  t= np.random.rand()

  if(i<2):
    angle= np.random.rand(ANGLE_DIM,1)
  else:
    x = int(ACT_GEN_WIDTH)
    l=np.linspace(0,1,x)

    candidate_array = np.zeros((x**ANGLE_DIM,INPUT_DIM))
    candidate_score = np.zeros(x**ANGLE_DIM)

    index = 0

    for v in itertools.product(l,repeat=ANGLE_DIM):
      candidate_array[index,:] = np.hstack((np.hstack((angle_history[:-1,:])),np.array(v)))
      #candidate_array[index,:] = np.hstack((present_angle,np.array(v)))
      reshape=np.reshape(candidate_array[index,:].T,[-1,INPUT_DIM])

      candidate_score[index] = neural.predict(reshape)
      index+=1

    angle = candidate_array[np.argmax(candidate_score),-ANGLE_DIM:]

  return t,np.reshape(angle,(-1,ANGLE_DIM))

def conv_angle(key_str,angle):
    key_list = ["HeadYaw","HeadPitch","LShoulderRoll", "RShoulderRoll","LShoulderPitch", "RShoulderPitch","LElbowRoll", "RElbowRoll","LElbowYaw", "RElbowYaw"]
    angle_array = np.array([[-2.0,2.0],[-0.6,0.5],[-0.3,1.3],[-1.3,0.3],[-2.0,2.0],[-2.0,2.0],[-1.5,0],[0,1.5],[-2.0,2.0],[-2.0,2.0]])
    index=key_list.index(key_str)
    a = abs(angle_array[index,0])+abs(angle_array[index,1])
    b = angle_array[index,0]
    return angle*a + b


def move(motion_proc,angle,t):

    speed = t

    #motion_proc.changeAngles(["HeadYaw", "HeadPitch"], [conv_angle("HeadYaw",angle[0,0]), conv_angle("HeadPitch",angle[0,1])], speed)
    motion_proc.changeAngles(["LShoulderRoll", "RShoulderRoll"], [conv_angle("LShoulderRoll",angle[0,0]), conv_angle("RShoulderRoll",angle[0,1])],speed )
    #motion_proc.changeAngles(["LShoulderPitch", "RShoulderPitch"], [conv_angle("LShoulderPitch",angle[0,0]), conv_angle("RShoulderPitch",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowRoll", "RElbowRoll"], [conv_angle("LElbowRoll",angle[0,0]), conv_angle("RElbowRoll",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowYaw", "RElbowYaw"], [conv_angle("LElbowYaw",angle[0,0]), conv_angle("RElbowYaw",angle[0,1])],speed)

def motion_nao(q_face):
  #neural = Neural("NAO",ANGLE_DIM*2,FACE_DIM,LEARN_NUM)
  neural = Neural("NAO",INPUT_DIM,FACE_DIM,LEARN_NUM)
  neural.get_model("new")

  face_history = np.zeros((LOOP,FACE_DIM))
  angle_history = np.zeros((LOOP,ANGLE_DIM))

  motion = ALProxy("ALMotion", HOST, PORT)
  present_angle = np.zeros(ANGLE_DIM)
  move(motion,np.reshape(present_angle,(-1,ANGLE_DIM)),1)

  loop_count = 0

  for i in range(TRIAL_NUM*LOOP):
    #t,angle = get_angle(present_angle,neural,i)
    t,angle = get_angle(angle_history,neural,i)
    #np.savetxt("angle.csv",np.reshape(np.array(angle),[1,2]),delimiter=",")
    np.savetxt("angle.csv",angle,delimiter=",")

    face_history[:-1,:] = face_history[1:,:]
    angle_history[:-1,:] = angle_history[1:,:]

    move(motion,angle,t)

    if(q_face.empty()==False):
      face=q_face.get()

      face_history[-1,:] = np.reshape(face,[1,FACE_DIM])
      angle_history[-1,:] = np.reshape(angle,[1,ANGLE_DIM])

      if(loop_count%LOOP==0):
        #learning(neural,np.hstack((present_angle,angle[0,:])),face)
        learning(neural,np.hstack((angle_history)),face)
      with open("learning.txt",'a') as f:
        f.write(str(angle[0,:])+","+str(face)+"\n")

    present_angle = np.reshape(angle,(ANGLE_DIM))

    time.sleep(t)
    loop_count+=1


if __name__ == '__main__':
  q_face= Queue.Queue()

  th_nao = threading.Thread(target=motion_nao,args=(q_face,))
  th_face = threading.Thread(target=get_face,args=(q_face,))

  th_face.setDaemon(True)

  th_nao.start()
  th_face.start()

  th_nao.join()

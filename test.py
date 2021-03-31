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

TRIAL_NUM = 15
HOST = "127.0.0.1"
PORT= 40165

ANGLE_DIM= 2
FACE_DIM= 1
LEARN_NUM = 5

ACT_GEN_WIDTH = 5

def learning(neural,present_angle,angle,face):
  neural.train(np.reshape(np.hstack((present_angle,angle)),(-1,ANGLE_DIM*2)),np.reshape(face,(-1,FACE_DIM)))  

def get_angle(present_angle,neural,i):
  t= np.random.rand()

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
      reshape=np.reshape(candidate_array[index,:].T,[-1,4])

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
  neural = Neural("NAO",ANGLE_DIM*2,FACE_DIM,LEARN_NUM)

  face_history = np.zeros((1,FACE_DIM))
  angle_history = np.zeros((1,ANGLE_DIM))

  motion = ALProxy("ALMotion", HOST, PORT)
  present_angle = np.zeros(ANGLE_DIM)
  move(motion,np.reshape(present_angle,(-1,ANGLE_DIM)),1)

  for i in range(TRIAL_NUM):
    t,angle = get_angle(present_angle,neural,i)
    #np.savetxt("angle.csv",np.reshape(np.array(angle),[1,2]),delimiter=",")
    np.savetxt("angle.csv",angle,delimiter=",")

    move(motion,angle,t)

    if(q_face.empty()==False):
      face=q_face.get()

      face_history = np.append(face_history,np.reshape(face,[1,FACE_DIM]),axis=0)
      angle_history = np.append(angle_history,np.reshape(angle,[1,ANGLE_DIM]),axis=0)

      print("face",face)

      learning(neural,present_angle,angle[0,:],face)

    present_angle = np.reshape(angle,(ANGLE_DIM))

    time.sleep(t)


if __name__ == '__main__':
  q_face= Queue.Queue()

  th_nao = threading.Thread(target=motion_nao,args=(q_face,))
  th_face = threading.Thread(target=get_face,args=(q_face,))

  th_face.setDaemon(True)

  th_nao.start()
  th_face.start()

  th_nao.join()

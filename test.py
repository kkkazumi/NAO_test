from naoqi import *
import time 
import datetime
import numpy as np
import threading

import Queue

from face_module import get_face
from neural_network import *

epsilon = 0.1
mu = 0.9
epoch = 10

TRIAL_NUM = 5
HOST = "127.0.0.1"
PORT= 38813

ANGLE_DIM= 2
FACE_DIM= 1

def learning(neural,angle,face):
  neural.train(angle,face,epsilon,mu,epoch)  

def get_angle(present_angle,neural):
  t= np.random.rand()
  angle = np.zeros(2)

  print("predict",neural.predict(angle))

  for i in range(present_angle.shape[0]):
    angle[i] = present_angle[i]+2*np.random.rand()-1.0
    if angle[i]>=1.0:
      angle[i] = 1.0
    elif angle[i]<=-1.0:
      angle[i] = -1.0
  return t,angle

def move(motion_proc,angle,t):

    speed = t
    motion_proc.setAngles(["HeadYaw", "HeadPitch"], [angle[0], angle[1]], speed)
    motion_proc.setAngles(["LShoulderRoll", "RShoulderRoll"], [angle[0], angle[1]],speed )
    motion_proc.setAngles(["LShoulderPitch", "RShoulderPitch"], [angle[0], angle[1]],speed)
    motion_proc.setAngles(["LElbowRoll", "RElbowRoll"], [angle[0], angle[1]],speed)
    motion_proc.setAngles(["LElbowYaw", "RElbowYaw"], [angle[0], angle[1]],speed)

def motion_nao(q_face):
  neural = Neural(ANGLE_DIM,2,FACE_DIM)

  face_history = np.zeros((1,FACE_DIM))
  angle_history = np.zeros((1,ANGLE_DIM))

  motion = ALProxy("ALMotion", HOST, PORT)
  present_angle = np.zeros(ANGLE_DIM)

  for i in range(TRIAL_NUM):
    t,angle = get_angle(present_angle,neural)
    #np.savetxt("angle.csv",np.reshape(np.array(angle),[1,2]),delimiter=",")
    np.savetxt("angle.csv",angle,delimiter=",")

    move(motion,angle,t)

    if(q_face.empty()==False):
      face=q_face.get()

    face_history = np.append(face_history,np.reshape(face,[1,FACE_DIM]),axis=0)
    angle_history = np.append(angle_history,np.reshape(angle,[1,ANGLE_DIM]),axis=0)

    learning(neural,angle_history,face_history)

    time.sleep(1)


if __name__ == '__main__':
  q_face= Queue.Queue()

  th_nao = threading.Thread(target=motion_nao,args=(q_face,))
  th_face = threading.Thread(target=get_face,args=(q_face,))

  th_face.setDaemon(True)

  th_nao.start()
  th_face.start()

  th_nao.join()

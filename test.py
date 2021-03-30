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

TRIAL_NUM = 5
HOST = "127.0.0.1"
PORT= 37441

ANGLE_DIM= 2
FACE_DIM= 1
LEARN_NUM = 5

ACT_GEN_WIDTH = 5


def learning(neural,angle,face):
  neural.train(angle,face,epsilon,mu,epoch)  

def get_angle(present_angle,neural):
  t= np.random.rand()

  x = int(ACT_GEN_WIDTH)
  l=np.linspace(0,1,x)

  candidate_array = np.zeros((x**ANGLE_DIM,ANGLE_DIM*2))
  candidate_score = np.zeros(x**ANGLE_DIM)

  index = 0

  for v in itertools.product(l,repeat=ANGLE_DIM):
    candidate_array[index,:] = np.hstack((present_angle,np.array(v)))
    reshape=np.reshape(candidate_array[index,:].T,[-1,4])

    candidate_score[index] = neural.predict(reshape)
    #candidate_score[index] = neural.predict(candidate_array[index,:].T)
    index+=1

  #input_data = np.hstack((present_angle,angle))

  #print("predict",neural.predict(angle))

  #for i in range(present_angle.shape[0]):
  #  angle[i] = present_angle[i]+2*np.random.rand()-1.0
  #  if angle[i]>=1.0:
  #    angle[i] = 1.0
  #  elif angle[i]<=-1.0:
  #    angle[i] = -1.0

  angle = candidate_array[np.argmax(candidate_score),-ANGLE_DIM:]
  print("best angle",angle)

  return t,angle

def move(motion_proc,angle,t):

    speed = t
    motion_proc.setAngles(["HeadYaw", "HeadPitch"], [angle[0], angle[1]], speed)
    motion_proc.setAngles(["LShoulderRoll", "RShoulderRoll"], [angle[0], angle[1]],speed )
    motion_proc.setAngles(["LShoulderPitch", "RShoulderPitch"], [angle[0], angle[1]],speed)
    motion_proc.setAngles(["LElbowRoll", "RElbowRoll"], [angle[0], angle[1]],speed)
    motion_proc.setAngles(["LElbowYaw", "RElbowYaw"], [angle[0], angle[1]],speed)

def motion_nao(q_face):
  neural = Neural("NAO",ANGLE_DIM*2,FACE_DIM,LEARN_NUM)

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

    #learning(neural,angle_history,face_history)

    time.sleep(1)


if __name__ == '__main__':
  q_face= Queue.Queue()

  th_nao = threading.Thread(target=motion_nao,args=(q_face,))
  th_face = threading.Thread(target=get_face,args=(q_face,))

  th_face.setDaemon(True)

  th_nao.start()
  th_face.start()

  th_nao.join()

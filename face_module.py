import numpy as np
import threading
import datetime

import Queue

import os

def dummy_face_eval(angle):
  delta_face = 3*angle[1]-angle[0]
  return delta_face

def dummy_face(angle_queue,present_face):
  face = present_face+dummy_face_eval(angle_queue)/10.0
  #face = present_face+np.random.rand()/10.0
  if face>= 1.0:
    face = 1.0
  elif face<=0.0:
    face = 0.0
  return face,datetime.datetime.now()

def get_face(q_face):
  face = 0

  #for t in range(TRIAL_NUM):
  while(True):
    ## for dummy face function
    if(os.path.exists("angle.csv")==True):
      if(os.stat("angle.csv").st_size==0):
        face = 0
      else:
        angle = np.loadtxt("angle.csv",delimiter=",")
        os.remove("angle.csv")

        face,timestamp=dummy_face(angle,face)
        q_face.put(face)
        #print("face",face,timestamp)

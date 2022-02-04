from naoqi import *
import time 
import datetime
import numpy as np

HOST = "127.0.0.1"
PORT= 41849

def conv_angle(key_str,angle):
    key_list = ["HeadYaw","HeadPitch","LShoulderRoll", "RShoulderRoll","LShoulderPitch", "RShoulderPitch","LElbowRoll", "RElbowRoll","LElbowYaw", "RElbowYaw"]
    angle_array = np.array([[-2.0,2.0],[-0.6,0.5],[-0.3,1.3],[-1.3,0.3],[-2.0,2.0],[-2.0,2.0],[-1.5,0],[0,1.5],[-2.0,2.0],[-2.0,2.0]])
    index=key_list.index(key_str)
    a = abs(angle_array[index,0])+abs(angle_array[index,1])
    b = angle_array[index,0]
    return angle*a + b

def nao_move(proxy,angle,t):
    speed = 1.0
    print("move",angle,t)
    #motion_proc.changeAngles(["HeadYaw", "HeadPitch"], [conv_angle("HeadYaw",angle[0,0]), conv_angle("HeadPitch",angle[0,1])], speed)
    proxy.changeAngles(["LShoulderRoll", "RShoulderRoll"], [conv_angle("LShoulderRoll",angle[0,0]), conv_angle("RShoulderRoll",angle[0,1])],speed )
    #motion_proc.changeAngles(["LShoulderPitch", "RShoulderPitch"], [conv_angle("LShoulderPitch",angle[0,0]), conv_angle("RShoulderPitch",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowRoll", "RElbowRoll"], [conv_angle("LElbowRoll",angle[0,0]), conv_angle("RElbowRoll",angle[0,1])],speed)
    #motion_proc.changeAngles(["LElbowYaw", "RElbowYaw"], [conv_angle("LElbowYaw",angle[0,0]), conv_angle("RElbowYaw",angle[0,1])],speed)

def nao_data():
  robot_name = "NAO"
  robot = ALProxy("ALMotion", HOST, PORT)
  robot_move_func = nao_move
  return robot_name,robot,robot_move_func


#import optuna

import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
#import scipy as sp
#from scipy.stats import pearsonr

from keras.utils import plot_model

import os

MID_UNIT = 5

class Neural:
  def __init__(self, name,input_dim,output_dim,epoch):
    self.name = name
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.epoch = epoch

  def get_model(self,mode,para_unit=MID_UNIT,para_act="relu"):

    if(mode=="present"):
      self.model = load_model(self.name+"_model.h5")
      #self.status = self.model.load_weights("ckpt")
    elif(mode=="new"):
    #  #define NN framework
      self.model = Sequential()
      self.model.add(Dense(para_unit, input_dim = self.input_dim, activation=para_act))
      self.model.add(Dense(self.output_dim, activation=para_act))

    if(os.path.exists("ckpt")==True):
      self.status = self.model.load_weights("ckpt")

  #def objective(self,trial):
  #  print("here")
  #  K.clear_session()
  #
  #  mid_units = trial.suggest_int("mid_units", 5, 100)

  #  activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

  #  #optimizer
  #  optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
    
  #  self.get_model("new")

  #  self.model.compile(optimizer=optimizer,
  #            loss="mean_squared_error",
  #            metrics=["accuracy"])

  #  print("before fit")
  #  history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=200, batch_size=128, validation_split=0.1)
  #  return 1 - history.history["val_accuracy"][-1]

  def train(self,x_train_data,y_train_data,mode,i):

    self.x_train = x_train_data
    self.y_train = y_train_data

    #study = optuna.create_study()
    #study.optimize(self.objective, n_trials=trial_number)

    #mid_units=study.best_params["mid_units"]
    #activation=study.best_params["activation"]
    #optimizer=study.best_params["optimizer"]

    mid_units=5
    activation = "sigmoid"
    optimizer = "adam"

    #best_params = ["mid_units:",str(mid_units),",activation:" ,activation,",optimizer:",optimizer]
    #path_w  = self.name+"_best_params.txt"
    #with open(path_w, mode='w') as f:
    #  f.writelines(best_params)

    #define NN framework

    self.get_model(mode,para_unit=mid_units,para_act=activation)
    print("before compile")
    plot_model(self.model, to_file='model_'+str(i)+'.png')

    self.model.compile(optimizer=optimizer,
      loss="mean_squared_error",
      metrics=["accuracy"])
    
    print("training data",self.x_train,self.y_train)

    train=self.model.fit(x=self.x_train, y=self.y_train, nb_epoch=self.epoch)
    self.model.save(self.name+"_model.h5")
    self.model.save_weights("ckpt")

    lossname = self.name+ "_loss.csv"
    np.savetxt(lossname,train.history['loss'])

  def predict(self,input_data):
    if(input_data.shape[1] != self.input_dim):
      print("input correct size data",input_data.shape[1],self.input_dim)
      output_data = None
    else:
      #plot_model(self.model, to_file='model.png',show_shapes=True)
      output_data= self.model.predict(input_data)
    return output_data

if __name__ == '__main__':
  name = "B0000"
  trial_number = 10
  x_train = np.tile(np.linspace(0,1,10),(4,1))
  y_train = np.linspace(0,0.5,10)*2+np.ones_like(np.linspace(0,1,10))*0.5
  neural = Neural(name,4,1,trial_number)
  #neural.train(x_train.T,y_train.T,"new",0)
  neural.get_model("new")
  x_test = np.array([[1,1,1,1]])
  print("return",neural.predict(x_test))

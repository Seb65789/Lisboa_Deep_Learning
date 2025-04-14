import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from src.utils import load_dataset
from src.plots import plot,plot_loss,plot_w_norm
import os 


# Importing the classes
#================================================================================================#

from src.linearmodels import Perceptron
from src.linearmodels import LogisticRegressionScratch

#================================================================================================#


def main():
    
    # Command line Arguments 
    #============================================================================================#

    arguments = argparse.ArgumentParser() # Creating the arguments

    arguments.add_argument('model',choices=['perceptron','mlp_scratch','log_reg_scratch','log_reg_torch','mlp_torch']) # Model choice

    arguments.add_argument('-epochs',default=20,type=int) # How many epochs

    arguments.add_argument("-data_path",default="src/data/intel_landscapes.npz",type=str) # The data

    arguments.add_argument("-lr",default=0.001,type=float)

    arguments.add_argument("-l2_penalty",default=0,type=float)

    opt = arguments.parse_args()

    #============================================================================================#

    
    # Load dataset
    #============================================================================================#

    data = load_dataset(opt.data_path,bias = opt.model=='mlp') # An object with : train , val , test keys
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    #============================================================================================#

    


    # Initialize the model
    #============================================================================================#
    
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1] # number of columns
    
    if opt.model == 'perceptron' :
      model = Perceptron(n_classes,n_features)


    elif opt.model == 'log_reg_scratch' :
      model = LogisticRegressionScratch(n_classes,n_features)


    # Tracking the metrics
    #============================================================================================#

    train_loss = []

    train_acc = []
    val_acc = []

    weight_norm = []

    #============================================================================================#

    
    # Training
    #============================================================================================#

    epochs = np.arange(1,opt.epochs+1)

    print("Initial train accuracy : {:.4f} | initial validation accuracy : {:.4f}"
          .format(model.evaluate(X_train,y_train),model.evaluate(X_val,y_val)))

    start = time.time()

    for epoch in epochs :
        print("Training epoch {}".format(epoch))

        # randomize the training order to generalize 
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        if opt.model != 'mlp' :
            model.train_epoch(X_train
                              ,y_train
                              ,lr=opt.lr
                              ,l2_penalty = opt.l2_penalty) 

        # Appending the metrics
        train_acc.append(model.evaluate(X_train,y_train))
        val_acc.append(model.evaluate(X_val,y_val))

        weight_norm.append(np.linalg.norm(model.W))

        # Printing 
        print("Training accuracy : {:.4f} | Validation accuracy : {:.4f}"
              .format(train_acc[-1],val_acc[-1]))
    
    end = time.time() - start
    minutes = int(end//60)
    seconds = int(end%60)

    #============================================================================================#


    # Testing
    #============================================================================================#

    with open(f"results/{opt.model}-results.txt","a") as f:
       if opt.model == 'perceptron':
          f.write("\nTraining took {}:{} - Final test accuracy {:.4f} " \
          "\n Parameters : \n\t epochs :{}".format(minutes,seconds,model.evaluate(X_test,y_test),opt.epochs))
       elif opt.model == 'log_reg_scratch' or opt.model == 'log_reg_torch':
          f.write("\nTraining took {}:{} - Final test accuracy {:.4f} - Weight's norm : {}" \
          "\n Parameters : \n\t -epochs :{} \n\t -learning_rate : {} \n\t -regularization : {}".format(minutes,seconds,model.evaluate(X_test,y_test),weight_norm[-1],opt.epochs,opt.lr,opt.l2_penalty))
    print("Training took {}:{}".format(minutes,seconds))
    print("Final test accuracy {:.4f}".format(model.evaluate(X_test,y_test)))
    
    #============================================================================================#

    if not(os.path.isdir("results/")) :
      os.makedirs("results/")  
      print("Results directory created")


    # Plots
    #============================================================================================#

    plot(epochs,train_acc,val_acc,filename=f"results/{opt.model} - accs.pdf")
    if opt.model == 'mlp' :
      plot_loss()
    elif opt.model == 'log_reg_scratch' or opt.model == 'log_reg_torch' :
      plot_w_norm(epochs,weight_norm,filename=f"results/{opt.model} - w_norms.pdf")

        

if __name__ == '__main__':
    main()
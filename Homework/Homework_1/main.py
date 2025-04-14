import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from src.utils import load_dataset
import os 
# Importing the classes
#================================================================================================#

from src.linearmodels import Perceptron

#================================================================================================#


def main():
    
    # Command line Arguments 
    #============================================================================================#

    arguments = argparse.ArgumentParser() # Creating the arguments

    arguments.add_argument('model',choices=['perceptron','mlp','linear_regression']) # Model choice

    arguments.add_argument('-epochs',default=20,type=int) # How many epochs

    arguments.add_argument("-data_path",default="Homework/Homework_1/src/data/intel_landscapes.npz",type=str) # The data
    
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
    
    model = Perceptron(n_classes,n_features)



    # Tracking the metrics
    #============================================================================================#

    train_loss = []

    train_acc = []
    val_acc = []

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

        model.train_epoch(X_train,y_train) # Training

        # Appending the metrics
        train_acc.append(model.evaluate(X_train,y_train))
        val_acc.append(model.evaluate(X_val,y_val))

        # Printing 
        print("Training accuracy : {:.4f} | Validation accuracy : {:.4f}"
              .format(train_acc[-1],val_acc[-1]))
    
    end = time.time() - start
    minutes = int(end//60)
    seconds = int(end%60)

    #============================================================================================#


    # Testing
    #============================================================================================#

    print("Training took {}:{}".format(minutes,seconds))
    print("Final test accuracy {:.4f}".format(model.evaluate(X_test,y_test)))
    
    #============================================================================================#

    if not(os.path.isdir("Homework/Homework_1/results/")) :
          os.makedirs("Homework/Homework_1/results/")  
          print("Results directory created")


    # Plots
    #============================================================================================#

    plt.plot(epochs, train_acc,label = 'training accuracy')
    plt.plot(epochs,val_acc,label='validation accuracy')
    plt.title(f"{opt.model} Training and Validation accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("Homework/Homework_1/results/Train and validation accuracies - {}_{}".format(opt.model,opt.epochs))

if __name__ == '__main__':
    main()
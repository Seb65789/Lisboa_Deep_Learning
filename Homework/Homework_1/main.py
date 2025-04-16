import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from src.utils import load_dataset,configure_seed,ClassificationDataset
from src.plots import plot,plot_loss,plot_w_norm,plot_torch
import os 
import torch
from torch import nn
from src.models_torch import evaluate,predict,train_batch
from torch.utils.data import DataLoader



# Importing the classes
#================================================================================================#

from src.linearmodels_scratch import Perceptron
from src.linearmodels_scratch import LogisticRegressionScratch
from src.mlp_scratch import MultiLayerPerceptronScratch
from src.linearmodels_torch import LinearModel

#================================================================================================#


def main():
    
    # On GPU

    device = torch.cuda.device("cuda" if torch.cuda.is_available() else 'cpu')

    print("Running now on ", torch.cuda.get_device_name(0))
    
    # Command line Arguments 
    #============================================================================================#

    arguments = argparse.ArgumentParser() # Creating the arguments

    arguments.add_argument('model',choices=['perceptron','mlp_scratch','log_reg_scratch','log_reg_torch','mlp_torch','mlp_scratch']) # Model choice

    arguments.add_argument('-epochs',default=20,type=int) # How many epochs

    arguments.add_argument("-data_path",default="./src/data/intel_landscapes.npz",type=str) # The data

    arguments.add_argument("-lr",default=0.001,type=float)

    arguments.add_argument("-l2_penalty",default=0,type=float)

    arguments.add_argument("-hidden_size",default=100,type=int)

    arguments.add_argument("-optimizer",choices=['sgd','adam'],type=str)

    arguments.add_argument("-activation",choices=['tanh','relu'],type=str)

    arguments.add_argument("-momentum",default=0,type=float)
    
    arguments.add_argument("-dropout",default=0,type=float)
    
    arguments.add_argument("-layers",default=2,type=int)
    
    arguments.add_argument("-batch_size",default=64,type=int)

    opt = arguments.parse_args()

    #============================================================================================#

    
    

    configure_seed(seed=42)

    train_losses = []
    val_losses = []
    val_accs = []

    if opt.model == 'mlp_torch' or opt.model == 'log_reg_torch' :


    # Load dataset
    #============================================================================================#
      data = load_dataset(opt.data_path)
      dataset = ClassificationDataset(data) # Creates tensor from data
      train_dataloader = DataLoader(
          dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
      val_X, val_y = dataset.val_X, dataset.val_y
      X_test, y_test = dataset.test_X, dataset.test_y

      n_classes = torch.unique(dataset.y).shape[0] 
      n_feats = dataset.X.shape[1]

      print(f"N features: {n_feats}")
      print(f"N classes: {n_classes}")

    #============================================================================================#


    # Initialize the model
    #============================================================================================#
      if opt.model == 'mlp_torch':
        raise NotImplementedError
    
      else:
        model = LinearModel(n_classes=n_classes,n_features=n_feats)

      # Optimizer
      optims = {"adam": torch.optim.Adam,"sgd":torch.optim.SGD}

      optimizer = optims[opt.optimizer]

      optimizer = optimizer(model.parameters()
                            ,lr = opt.lr
                            ,momentum = opt.momentum
                            ,weight_decay = opt.l2_penalty) 

      # Criterion
      criterion = nn.CrossEntropyLoss()

      # training
      epochs = torch.arange(1,opt.epochs+1)

      start = time.time()

      print("Initial validation accuracy {:.4f}".format(evaluate(model,val_X,val_y,criterion)[1]))

      for epoch in epochs :
        print(f"Training epoch {epoch}")
        epoch_train_losses = []
        for X_batch,y_batch in train_dataloader:
          loss = train_batch(model,X_batch,y_batch,optimizer,criterion)
          epoch_train_losses.append(loss)
        
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        val_loss, val_acc = evaluate(model,val_X,val_y,criterion)

        print("Train loss: {:.4f} | Validation loss: {:.4f} | Validation accuracy: {:.4f}".format(epoch_train_loss,val_loss,val_acc))

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

      end = time.time() - start
      minutes = int(end//60)
      seconds = int(end%60)

      print("Training took {}:{}".format(minutes,seconds))
      _, test_acc = evaluate(model,X_test,y_test,criterion)
      print("Final test accuracy: {:.4f}".format(test_acc))

      if not(os.path.isdir(f"results/{opt.model}/")) :
        os.makedirs(f"results/{opt.model}/")  
        print("Results directory created")

      with open(f"results/{opt.model}/results.txt","a") as f:
        if opt.model == 'log_reg_torch':
            f.write("\nTraining took {}:{} - Final test accuracy {:.4f}" \
            "\n Parameters : \n\t -epochs : {} \n\t -learning_rate : {} \n\t -regularization : {}"\
              " \n\t -Optimizer : {} \n\t -batch_size : {}".format(minutes,seconds,test_acc,opt.epochs,opt.lr,opt.l2_penalty,opt.optimizer,opt.batch_size))
        elif opt.model == 'mlp_torch':
            f.write("\nTraining took {}:{} - Final test accuracy {:.4f}" \
            "\n Parameters : \n\t - hidden_size : {} \n\t -layers : {}\n\t -epochs :{} \n\t -learning_rate : {} \n\t -regularization : {}"
              " \n\t -Optimizer : {} \n\t -batch_size : {} \n\t -dropout : {}".format(minutes,seconds,test_acc,opt.hidden_size,opt.layers,opt.epochs,opt.lr,opt.l2_penalty,opt.optimizer,opt.batch_size, opt.dropout))
      
      
      #============================================================================================#

      if not(os.path.isdir("results/")) :
        os.makedirs("results/")  
        print("Results directory created")


    # Plots
    #============================================================================================#


      # plot
      if opt.model == "log_reg_torch":
          config = (
              f"batch-{opt.batch_size}-lr-{opt.lr}-epochs-{opt.epochs}-"
              f"l2-{opt.l2_penalty}-opt-{opt.optimizer}"
          )
      else:
          config = (
              f"batch-{opt.batch_size}-lr-{opt.lr}-epochs-{opt.epochs}-"
              f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_penalty}-"
              f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}-mom-{opt.momentum}"
          )

      losses = {
          "Train Loss": train_losses,
          "Valid Loss": val_losses,
      }

      plot_torch(epochs, losses, filename=f'results/{opt.model}/-training-loss-{config}.pdf')
      accuracy = { "Valid Accuracy": val_accs }
      plot_torch(epochs, accuracy, filename=f'results/{opt.model}/-validation-accuracy-{config}.pdf')
      
    #============================================================================================#

    else :
        
    # Load dataset
    #============================================================================================#

      data = load_dataset(opt.data_path,bias = opt.model=='mlp_scratch') # An object with : train , val , test keys
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

      elif opt.model == 'mlp_scratch':
        model = MultiLayerPerceptronScratch(n_classes,n_features,opt.hidden_size)
 
    #============================================================================================#


    # Tracking the metrics
    #============================================================================================#

      train_acc = []

      weight_norm = []

    #============================================================================================#

    
    # Training
    #============================================================================================#

      epochs = np.arange(1,opt.epochs+1)

      print("Initial train accuracy : {:.4f} | initial validation accuracy : {:.4f}"
            .format(model.evaluate(X_train,y_train),model.evaluate(X_val,y_val)))

      start = time.time()

      #print(X_train.shape)

      for epoch in epochs :
          print("Training epoch {}".format(epoch))

          # randomize the training order to generalize 
          train_order = np.random.permutation(X_train.shape[0])
          X_train = X_train[train_order]
          y_train = y_train[train_order]

          if opt.model != 'mlp_scratch' and opt.model != 'mlp_torch' :
              model.train_epoch(X_train
                                ,y_train
                                ,lr=opt.lr
                                ,l2_penalty = opt.l2_penalty) 
          else :
            loss = model.train_epoch(X_train,
                              y_train,
                              lr = opt.lr)

          # Appending the metrics
          train_acc.append(model.evaluate(X_train,y_train))
          val_accs.append(model.evaluate(X_val,y_val))
          train_losses.append(loss)
          if opt.model == 'log_reg_scratch' or opt.model == 'log_reg_torch':
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
        elif opt.model == 'mlp_scratch' or opt.model == 'mlp_torch':
            f.write("\nTraining took {}:{} - Final test accuracy {:.4f}" \
            "\n Parameters : \n\t -epochs :{} \n\t -learning_rate : {}".format(minutes,seconds,model.evaluate(X_test,y_test),opt.epochs,opt.lr))
        
            
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

      #============================================================================================#

if __name__ == '__main__':
  main()
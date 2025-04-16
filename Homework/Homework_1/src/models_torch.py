
import torch

def predict(model,X):
    scores = model(X)
    y_pred = torch.softmax(scores,dim=1)
    return y_pred 

def evaluate(model,X,y_true,criterion):
    raise NotImplementedError

def train_batch(model,X,y,optimizer, criterion,**kwargs) :
    raise NotImplementedError
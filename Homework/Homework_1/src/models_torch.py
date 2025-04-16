
import torch

def predict(model,X):
    scores = model(X)
    y_pred = torch.softmax(scores,dim=1)
    return y_pred 

def evaluate(model,X,y_true,criterion):
    model.eval()
    scores = model(X)
    
    y_pred = scores.argmax(dim=-1)

    loss = criterion(scores,y_true)
    loss = loss.item()
    
    model.train()
    return loss, (y_true == y_pred).sum().item()/float(y_true.shape[0])

def train_batch(model,X,y,optimizer, criterion,**kwargs) :
    optimizer.zero_grad() # Reset the optimizer for the new batch
    # prediction
    out = model(X,**kwargs)
    loss = criterion(out,y) # Compute the loss between pred and true
    loss.backward() # Computes dloss/dx
    optimizer.step() # Compute for SGD x -= lr*x.grad
    return loss.item() # return the value of the loss
from torch.nn import Module

class Model(Module):
    def __init__(self,n_classes,n_features,**kwargs):
        super().__init__()
    
    def evaluate(model,X,y):
        y_pred = model.predict()
        n_corrects = (y == y_pred).sum()
        return n_corrects/y.shape[0]
    
    def predict(self,X):
        scores = self(X)

    
    def train_batch(self,X,y,optimizer,criterion,**kwargs):
        raise NotImplementedError
    
    
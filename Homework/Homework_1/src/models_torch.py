class Model(object):
    def __init__(self,n_classes,n_features,**kwargs):
        raise NotImplementedError
    
    def evaluate(model,X,y):
        raise NotImplementedError
    
    def predict(self,X):
        raise NotImplementedError
    
    def train_batch(self,X,y,optimizer,criterion,**kwargs):
        raise NotImplementedError
    
    
import numpy as np

class LinearModel(object) :
    def __init__(self,n_classes,n_features,**kwargs):
        # Initialize the weights
        self.W = np.zeros((n_classes,n_features))

    def update_weights(self,x_i,y_i,**kwargs):
        raise NotImplementedError
    
    def train_epoch(self,X,y,**kwargs):
        for x_i,y_i in X,y :
            self.update_weights(x_i,y_i,**kwargs)
    
    def predict(self,X):
        # multiply the weights by the input
        scores = np.dot(self.W,X.T) #  produces n_classes x n_examples
        pred_labels = scores.argmax(axis=0)
        return pred_labels

    def evaluate(self,X,y):
        y_pred = self.predict(X)
        return (y == y_pred).sum()/y.shape[0]
    



class Perceptron(LinearModel):
    def update_weights(self, x_i, y_i, **kwargs):
        # prediction on the sample
        y_pred = np.argmax(np.dot(self.W,x_i),axis=0) # n_classses, n_features x n_features = n_classes
        
        if (y_i != y_pred):
            self.W[y_i] += x_i # more weight to the right class
            self.W[y_pred] -= x_i # less weight to the wrong class
        


        
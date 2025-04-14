import numpy as np


class LinearModel(object) :
    def __init__(self,n_classes,n_features,**kwargs):
        # Initialize the weights
        self.W = np.zeros((n_classes,n_features))

    def update_weights(self,x_i,y_i,**kwargs):
        raise NotImplementedError
    
    def train_epoch(self,X,y,**kwargs):
        for x_i,y_i in zip(X,y) :
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
    def update_weights(self, x_i, y_i,**kwargs):
        # prediction on the sample
        y_pred = np.argmax(np.dot(self.W,x_i),axis=0) # n_classses, n_features x n_features = n_classes
        
        if (y_i != y_pred):
            self.W[y_i] += x_i # more weight to the right class
            self.W[y_pred] -= x_i # less weight to the wrong class
        



class LogisticRegressionScratch(LinearModel):
    def update_weights(self, x_i, y_i, lr = 0.001,l2_penalty = 0,**kwargs):
        scores = np.dot(self.W, x_i) # softmax in theory but we obtain the same results 
        probs = np.exp(scores)/np.sum(np.exp(scores))
        
        # gradient of the loss fonction
        grad_loss = probs #delta of L(W,(x_i,y_i))
        grad_loss[y_i] -= 1 # compute grad_loss - e_y which is [0,0,...,1,0,0,0] is the y_i position

        # Update of the weights
        self.W += -lr*(np.outer(grad_loss,x_i) + l2_penalty*self.W) 
        # outer to get a n_classes x n_features matrix to substract the weigths by



        
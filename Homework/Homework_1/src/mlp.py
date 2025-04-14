import numpy as np

class MultiLayerPerceptron(object):

    def __init__(self,n_classes,n_features,hidden_size):
        # Creating the weights
        self.W1 = np.random.normal(0.1,0.1**2,(hidden_size,n_features))
        self.b1 = np.zeros((hidden_size,1))
        self.W2 = np.random.normal(0.1,0.1**2,(n_classes,hidden_size))
    
    def fprog(self,X):
        
        # hidden layer pre-activation
        z1 = np.dot(self.W1,X.T) 
        # Activation function applied
        self.h = np.maximum(0,z1)

        z_out = np.dot(self.W2,self.h)
        # To avoid gradient vanishing
        z_out = z_out - np.max(z_out,axis=0,keepdims=True)   
        
        # Apply the cross-entropy loss -> softmax
        self.out = np.exp(z_out)/np.sum(np.exp(z_out),axis=0,keepdims=True)

    def bprog(self,x_i,y_i,lr):
        raise NotImplementedError

    def predict(self,X):
        # Compute through the mlp
        self.h = np.maximum(0,np.dot(self.W1,X.T) + self.b1)
        out = np.exp(np.dot(self.W2,self.h))/np.sum(np.exp(np.dot(self.W2,self.h)),axis=0,keepdims=True)
        return out.argmax(axis=0)
    
    def compute_loss(self,y_true):
        # one hot encoding of y_true
        y_true_hot = np.zeros(self.out.shape[0])
        y_true_hot[y_true] = 1

        self.out = np.clip(self.out, 1e10-14 , 1-1e10-14) # avoiding log(0)

        # loss computing
        return -np.sum(y_true_hot*np.log(self.out))
    




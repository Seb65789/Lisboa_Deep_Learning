import numpy as np

class MultiLayerPerceptronScratch(object):

    def __init__(self,n_classes,n_features,hidden_size):
        # Creating the weights
        self.W1 = np.random.normal(0.1,0.1**2,(hidden_size,n_features))
        #print(self.W1)
        self.b1 = np.zeros((hidden_size,1))
        self.W2 = np.random.normal(0.1,0.1**2,(n_classes,hidden_size))
    
    def fprop(self,X):
        #print(X)
        # hidden layer pre-activation
        z1 = np.dot(self.W1,X) + self.b1
        #print(self.W1)
        # Activation function applied
        self.h1 = np.maximum(0,z1)
        #print(self.h1)
        z_2 = np.dot(self.W2, self.h1)
        #print(z_2)
        z_2_shiftted = z_2 - np.max(z_2,axis=0,keepdims=True)

        # Apply the cross-entropy loss -> softmax
        self.out = np.exp(z_2_shiftted)/np.sum(np.exp(z_2_shiftted),axis=0,keepdims=True)
        self.out = np.clip(self.out, 1e-12, 1 - 1e-12)

    def bprop(self,x_i,y_i,lr):
        
        # one hot encoded
        d_out = self.out - y_i

        # application of algorithm
        # l = 2
        dW2 = np.dot(d_out,self.h1.T)
        db2 = d_out
        dh1 = np.dot(self.W2.T,d_out)
        dz1 = dh1 * (self.h1 >0) # derivate of relu fonction applied on z1

        # l=1
        dW1 = np.dot(dz1,x_i.T)
        db1 = dz1

        # Update
        self.W1 -= lr*dW1
        self.b1 -= lr*db1
        self.W2 -= lr*dW2

    def predict(self,X):
        # Compute through the mlp
        self.h1 = np.maximum(0,np.dot(self.W1,X.T) + self.b1)
        z = np.dot(self.W2, self.h1)
        z -= np.max(z, axis=0, keepdims=True)  # Stabilisation numérique
        out = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        return out.argmax(axis=0)
    
    def evaluate(self,X,y):
        # predictions    
        y_pred = self.predict(X)
        return (y == y_pred).sum()/y.shape[0]
    
    def compute_loss(self,y_true):
        # one hot encoding of y_true
        y_true_hot = np.zeros_like(self.out)
        y_true_hot[y_true,np.arange(y_true.size)] = 1
        #print(y_true_hot.shape)

        self.out = np.clip(self.out, 1e-10 , 1 - 1e-10) # avoiding log(0)
        
        # loss computing
        loss = - np.sum(y_true_hot*np.log(self.out))
        return loss
    
    def train_epoch(self,X,y,lr = 0.001,**kwargs) :
        total_loss = 0
        #print(X.shape)

        for x_i,y_i in zip(X,y) :
            # forward
            
            self.fprop(x_i.reshape(-1,1)) # reshaped as a column vector

            #one hot encoding
            y_true_hot = np.zeros((self.out.shape[0],1)) # to be column
            y_true_hot[y_i] = 1

            total_loss += self.compute_loss(y_i)

            #bprog
            self.bprop(x_i.reshape(-1,1),y_true_hot,lr)
            
        return total_loss / X.shape[0]




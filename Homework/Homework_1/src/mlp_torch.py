import torch 
from torch import nn
import ast

class MLP(nn.Module):
    def __init__(self,n_classes,n_features,hidden_size,layers,activation_type,dropout,**kwargs):
        super().__init__()
        # To choose the activation
        activations = {
            'tanh':nn.Tanh(),
            'relu':nn.ReLU(),
        }

        # Activation
        activation = activations[activation_type]

        drop_layer = nn.Dropout(dropout)


        # Creates the list out the input sizes automaticly from hidden_size and layers
        if isinstance(hidden_size,int) :
            in_sizes = [n_features] + [hidden_size] * layers # [n_features,hidden_size, ... ,hidden_size]
            out_sizes = [hidden_size]*layers + [n_classes] # [hidden_size, ... ,hidden_size,n_classes]

        else :
            in_sizes = [n_features] + ast.literal_eval(hidden_size) # same result but for hidden_size being a list
            out_sizes = ast.literal_eval(hidden_size) + [n_classes]


        layers = [layers.append(nn.Linear(in_size,out_size)),layers.append(activation),layers.append(drop_layer)
                  for in_size, out_size in zip(in_sizes[:-1],out_sizes[:-1])]
    
        layers.append(nn.Linear(in_sizes[-1],out_sizes[-1]))
        self.feedforward = nn.Sequential(*layers) # *to unpack the list
    
    def forward(self,x,**kwargs):
        return self.feedforward(x)
import numpy as np
import torch 
from torch.nn import Module
from torch import nn

class LinearModel(Module) :
    def __init__(self,n_classes,n_features,**kwargs) :
        # Initialize based on Model
        super().__init__()
        # Create the linear layer that will learn the weights from input
        self.linear = nn.Linear(in_features=n_features,out_features=n_classes)

    def forward(self,x,**kwargs):
        # The forward only needs the scores 
        return self.linear(x)
    
    def evaluate(model, X, y):
        return super().evaluate(X, y)

    def train_batch(self, X, y, optimizer, criterion, **kwargs):
        return super().train_batch(X, y, optimizer, criterion, **kwargs)


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def ComputeGradsWithTorch(X, y, network_params, lam = 0.01):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X.T)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['weights'], requires_grad=True)
    b = torch.tensor(network_params['bias'], requires_grad=True)    
    
    N = X.shape[0]
    
    scores = torch.matmul(W, Xt)  + b

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    # compute costs with regularization
    cost = loss + lam * torch.sum(torch.multiply(W, W))

    # compute the backward pass relative to the cost and the named parameters
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['weights'] = W.grad.numpy()
    grads['bias'] = b.grad.numpy()

    return cost, grads    

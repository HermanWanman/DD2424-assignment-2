import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params, lam=0):
    # Enforce float64 (double) to exactly match NumPy's precision
    Xt = torch.from_numpy(X).double() 
    
    # Convert index arrays to PyTorch LongTensors for safe indexing
    n = X.shape[1]
    y_tensor = torch.tensor(y, dtype=torch.long)
    idx_tensor = torch.arange(n, dtype=torch.long)
    
    L = len(network_params)
    W = []
    b = []
    for i in range(L):
        # Initialize tensors with float64 precision
        W.append(torch.tensor(network_params[i]['weights'], requires_grad=True, dtype=torch.double))
        b.append(torch.tensor(network_params[i]['bias'], requires_grad=True, dtype=torch.double))
        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)
    
    #### BEGIN your code ###########################
    s = W[0] @ Xt + b[0]
    s = apply_relu(s)
    for i in range(1, L):
        s = W[i] @ s + b[i]
        if i < L - 1:  
            s = apply_relu(s)
    scores = s
    #### END of your code ###########################
    
    P = apply_softmax(scores)
    
    # Compute the mean cross-entropy loss
    loss = torch.mean(-torch.log(P[y_tensor, idx_tensor]))
    
    # Compute the L2 regularization cost
    l2_reg = 0
    for w_mat in W:
        l2_reg += torch.sum(w_mat * w_mat)
        
    # Final cost is loss + lambda * L2_penalty
    cost = loss + lam * l2_reg
    
    # compute the backward pass relative to the cost
    cost.backward()

    grads = []
    for i in range(L):
        grads.append({"weights_grad": W[i].grad.numpy(), "bias_grad": b[i].grad.numpy()})
        
    return grads

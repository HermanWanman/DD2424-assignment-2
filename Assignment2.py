import copy
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch

matplotlib.use('Qt5Agg') 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def LoadBatch(batch_number):
    filename = f'./Datasets/cifar-10-batches-py/data_batch_{batch_number}'
    if batch_number == 5:
        filename = f'./Datasets/cifar-10-batches-py/test_batch'
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes') 
    
    imagePixelData = data[b'data'].astype(np.float64) / 255.0 
    imageLabels = np.array(data[b'labels']) 
    
    oneHotRep = np.zeros((10, len(imageLabels))) 
    oneHotRep[imageLabels, np.arange(len(imageLabels))] = 1 
    
    return imagePixelData, oneHotRep, imageLabels

def computeMeanStd(train_data):
    """Computes the mean and std on the training data ONLY."""
    mean_X = np.mean(train_data, axis=0, keepdims=True)
    std_X = np.std(train_data, axis=0, keepdims=True)
    return mean_X, std_X

def normalizeData(data, mean_X, std_X):
    """Applies pre-computed mean and std to normalize data."""
    return (data - mean_X) / std_X

def softmax(x):
    z = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

def initializeWeights(k, d, seed=42):
    np.random.seed(seed) 
    weights = np.random.normal(0, 1/np.sqrt(d), (k, d))  
    return weights

def initializeBias(k):
    bias = np.zeros((k, 1)) 
    return bias

def initializeModel(dims, seed=42):
    model = [dict()] * len(dims) 
    for i, dim in enumerate(dims):
        if dim[0] <= 0 or dim[1] <= 0:
            raise ValueError("Dimensions must be positive integers.")
        
        weights = initializeWeights(dim[0], dim[1], seed) 
        bias = initializeBias(dim[0]) 
        model[i] = {"weights": weights, "bias": bias} 
    return model

def applyLayer(pixelData, layer, apply_relu=True):
    weights = layer["weights"] 
    bias = layer["bias"] 
    
    z = np.matmul(weights, pixelData) + bias 
    
    if apply_relu:
        a = np.maximum(z, 0) 
    else:
        a = softmax(z) 

    return z, a 

def applyNetwork(pixelData, model):
    activations = pixelData.T 
    z_values = [] 
    a_values = [activations] 
    
    for i, layer in enumerate(model):

        is_hidden_layer = (i < len(model) - 1)
        z, a = applyLayer(activations, layer, apply_relu=is_hidden_layer)  
        z_values.append(z)
        a_values.append(a)
        activations = a
    
    return z_values, a_values

def lcross(labels, finalPredictions, onehot: bool = False):
    epsilon = 1e-15 
    safe_outputProbs = np.clip(finalPredictions, epsilon, 1.0 - epsilon) 
    
    if onehot:
        lcross_val = -np.sum(labels * np.log(safe_outputProbs), axis=0) 
    else:

        lcross_val = -np.log(safe_outputProbs[labels, np.arange(labels.shape[0])]) 
    return lcross_val  

def computeLoss(z_values, a_values, model, labels, l, onehot: bool = False):

    sum1 = np.sum(lcross(labels, a_values[-1], onehot)) 
    
    sum2 = 0
    for layer in model:
        sum2 += np.sum(layer["weights"] ** 2) 

    N = a_values[-1].shape[1] 
    total_loss = (1/N) * sum1 + l * sum2  
    return total_loss 

def getPredictedLabels(z_values, a_values):
    predicted_labels = np.argmax(a_values[-1], axis=0) 
    return predicted_labels

def computeAccuracy(predicted_labels, labels):
    accuracy = np.mean(predicted_labels == labels) 
    return accuracy 

def BackwardPass(z_values, a_values, model, labels, l):
    L = len(model) 
    N = a_values[-1].shape[1] 
    
    grads = [None] * L 
    
    # LAYER L-1 (Output layer with softmax)
    dL_dz = a_values[-1].copy() 
    dL_dz[labels.flatten(), np.arange(N)] -= 1 
    
    dL_dw = (1/N) * np.matmul(dL_dz, a_values[-2].T) + 2 * l * model[-1]["weights"]
    dL_db = (1/N) * np.sum(dL_dz, axis=1, keepdims=True)
    grads[-1] = {"weights": dL_dw, "bias": dL_db}
    
    # BACKPROPAGATE through earlier layers
    for i in range(L-2, -1, -1):
        dL_da = np.matmul(model[i+1]["weights"].T, dL_dz)
        dL_dz = dL_da * (z_values[i] > 0)
        
        dL_dw = (1/N) * np.matmul(dL_dz, a_values[i].T) + 2 * l * model[i]["weights"]
        dL_db = (1/N) * np.sum(dL_dz, axis=1, keepdims=True)
        grads[i] = {"weights": dL_dw, "bias": dL_db}
    
    return grads

def relativeError(grad1, grad2, eps=1e-8):
    nominator = np.abs(grad1 - grad2)
    if np.sum(np.abs(grad1)) + np.sum(np.abs(grad2)) < eps:
        return np.abs(grad1 - grad2)/eps
    else:
        denominator = np.abs(grad1) + np.abs(grad2)
        return nominator/denominator  

def miniBatchGradientDescent(
        X_train, labels_train, X_val, labels_val, n_batch, 
         n_epochs, model, lam, learningRateCalc="static", seed=42): 
    
    n = X_train.shape[0] 
    model_trained = copy.deepcopy(model) 

    train_costs, val_costs = [], []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    update_steps = []

    for epoch in range(n_epochs):
        rng = np.random.RandomState(seed + epoch)
        shuffled_indices = rng.permutation(n) 

        for i in range(n // n_batch):
            i_start = i * n_batch 
            i_end = (i+1) * n_batch 
            inds = np.arange(i_start, i_end) 

            t = epoch * n//n_batch + i # update step, saved for plotting

            if learningRateCalc == "cyclical":
                learningRate = computeCyclicalLearningRate(eta_min=1e-5, eta_max=1e-1, n_s=500, t=t)
            else:
                learningRate = 1e-3  # static learning rate

            x_batch = X_train[shuffled_indices[inds], :] 
            y_batch = labels_train[shuffled_indices[inds]] 
            
            z_batch, a_batch = applyNetwork(x_batch, model_trained) 
            grads = BackwardPass(z_batch, a_batch, model_trained, y_batch, l=lam) 
            
            for layer_idx in range(len(model_trained)):
                model_trained[layer_idx]["weights"] -= learningRate * grads[layer_idx]["weights"]
                model_trained[layer_idx]["bias"] -= learningRate * grads[layer_idx]["bias"]

        # Record the current update step (which is the total steps taken up to this point)
        current_step = (epoch + 1) * (n // n_batch)
        update_steps.append(current_step)

        # Evaluate Training Data
        z_train, a_train = applyNetwork(X_train, model_trained) 
        train_costs.append(computeLoss(z_train, a_train, model_trained, labels_train, l=lam)) 
        train_losses.append(computeLoss(z_train, a_train, model_trained, labels_train, l=0)) 
        train_accs.append(computeAccuracy(getPredictedLabels(z_train, a_train), labels_train))

        # Evaluate Validation Data
        z_val, a_val = applyNetwork(X_val, model_trained) 
        val_costs.append(computeLoss(z_val, a_val, model_trained, labels_val, l=lam))  
        val_losses.append(computeLoss(z_val, a_val, model_trained, labels_val, l=0)) 
        val_accs.append(computeAccuracy(getPredictedLabels(z_val, a_val), labels_val))

    return model_trained, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps 

def computeCyclicalLearningRate(eta_min, eta_max, n_s, t):
    l = t // (2 * n_s)
    if 2*l*n_s <= t <= (2*l+1)*n_s:
        learningRate = eta_min + ((t - 2*l*n_s) / n_s) * (eta_max - eta_min)
    else:        
        learningRate = eta_max - ((t - (2*l+1)*n_s) / n_s) * (eta_max - eta_min)
    return learningRate

def main():
    train_X, _, train_y = LoadBatch(1) 
    mean_X, std_X = computeMeanStd(train_X)
    train_X = normalizeData(train_X, mean_X, std_X) 
    print("Training data loaded and normalized.")

    val_X, _, val_y = LoadBatch(2) 
    val_X = normalizeData(val_X, mean_X, std_X) 
    print("Validation data loaded and normalized.")

    test_X, _, test_y = LoadBatch(5)
    test_X = normalizeData(test_X, mean_X, std_X)
    print("Test data loaded and normalized.")

    m = 50 
    dims = [[m, train_X.shape[1]],
            [len(np.unique(train_y)), m]]  
    model = initializeModel(dims) 
    print("Model initialized with weights and bias.")

    z_output, a_output = applyNetwork(train_X, model) 
    cross_entropy_loss = computeLoss(z_output, a_output, model, train_y, l=0.01) 
    print(f"Cross-entropy loss computed: {cross_entropy_loss}")

    get_predicted_labels = getPredictedLabels(z_output, a_output) 
    accuracy = computeAccuracy(get_predicted_labels, train_y) 
    print(f"Accuracy computed: {accuracy}")

    grads = BackwardPass(z_output, a_output, model, train_y, l=0.01) 
    print("Gradients computed.")

    torch_gradients = ComputeGradsWithTorch(train_X.T, train_y, model, lam=0.01)  
    tgW_0 = torch_gradients[0]['weights_grad'] 
    tgb_0 = torch_gradients[0]['bias_grad']  

    tgW_1 = torch_gradients[1]['weights_grad']  
    tgb_1 = torch_gradients[1]['bias_grad']  

    print(f"Relative error in gradients for layer 0 weights: {np.max(relativeError(grads[0]['weights'], tgW_0))}")
    print(f"Relative error in gradients for layer 0 bias: {np.max(relativeError(grads[0]['bias'], tgb_0))}")
    print(f"Relative error in gradients for layer 1 weights: {np.max(relativeError(grads[1]['weights'], tgW_1))}")
    print(f"Relative error in gradients for layer 1 bias: {np.max(relativeError(grads[1]['bias'], tgb_1))}")

    trained_model, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps = miniBatchGradientDescent(
        train_X, train_y, val_X, val_y, n_batch=100, learningRateCalc="cyclical", n_epochs=10, model=model, lam=0.01, seed=42)
    print("Mini-batch gradient descent completed.")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # Create 1 row with 3 columns

    # Cost Plot
    axs[0].plot(update_steps, train_costs, label='training cost', color='teal')
    axs[0].plot(update_steps, val_costs, label='validation cost', color='crimson')
    axs[0].set_xlabel('update step')
    axs[0].set_ylabel('cost')
    axs[0].set_title('Cost plot')
    axs[0].legend()

    # Loss Plot
    axs[1].plot(update_steps, train_losses, label='training loss', color='teal')
    axs[1].plot(update_steps, val_losses, label='validation loss', color='crimson')
    axs[1].set_xlabel('update step')
    axs[1].set_ylabel('loss')
    axs[1].set_title('Loss plot')
    axs[1].legend()

    # Accuracy Plot
    axs[2].plot(update_steps, train_accs, label='training accuracy', color='teal')
    axs[2].plot(update_steps, val_accs, label='validation accuracy', color='crimson')
    axs[2].set_xlabel('update step')
    axs[2].set_ylabel('accuracy')
    axs[2].set_title('Accuracy plot')
    axs[2].legend()

    plt.tight_layout() # fix layout
    plt.show()


if __name__ == "__main__":
    main()

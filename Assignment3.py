import copy
import time
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch, ComputePytorchGradientsConv

matplotlib.use('Qt5Agg') 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def LoadBatch(batch_number):
    filename = f'./Datasets/cifar-10-batches-py/data_batch_{batch_number}'
    if batch_number == -1:
        filename = f'./Datasets/cifar-10-batches-py/test_batch'
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes') 
    
    imagePixelData = data[b'data'].astype(np.float64) / 255.0 
    imageLabels = np.array(data[b'labels']) 
    
    oneHotRep = np.zeros((10, len(imageLabels))) 
    oneHotRep[imageLabels, np.arange(len(imageLabels))] = 1 
    return imagePixelData, oneHotRep, imageLabels

def debug_data_load():
    debug_file = './debug_conv_info.npz'
    load_data = np.load(debug_file)
    X = load_data['X']
    Fs = load_data['Fs']
    n = X.shape[1]
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

    true_convolutions = load_data['conv_outputs']
    true_labels = load_data['Y']
    return X_ims, Fs, true_convolutions, true_labels

def computeMeanStd(train_data):
    """Computes the mean and std on the training data ONLY."""
    mean_X = np.mean(train_data, axis=0, keepdims=True)
    std_X = np.std(train_data, axis=0, keepdims=True)
    return mean_X, std_X

def normalizeData(data, mean_X, std_X):
    """Applies pre-computed mean and std to normalize data."""
    return (data - mean_X) / std_X

def seq_convolutional_layer_calculation(X_ims, Fs, num_filters, stride = 4):
    n = X_ims.shape[3]
    image_size = X_ims.shape[0]
    num_rows = int(image_size / stride)
    num_cols = int(image_size / stride)
    computed_conv_outputs = np.zeros((32//stride, 32//stride, num_filters, n))
    for i in range(n):
        for row in range(num_rows):
            for col in range(num_cols):
                current_patch = X_ims[row*int(32/num_rows):(row+1)*int(32/num_rows), col*int(32/num_cols):(col+1)*int(32/num_cols), :, i] # extract the patch from input data
                for k in range(num_filters):
                    current_filter = Fs[:,:,:,k] # extract the filter
                    computed_conv_outputs[row,col,k,i] = np.sum(np.multiply(current_patch , current_filter))
    return computed_conv_outputs

def MX_initialization(X_ims, stride = 4):
    """"Initializes the MX matrix for convolutional layer calculations. Requires square images and stride that divides the image size."""
    n = X_ims.shape[3]
    image_size = X_ims.shape[0]
    num_rows = int(image_size / stride)
    num_cols = int(image_size / stride)
    MX = np.zeros((num_rows * num_cols, 3*stride*stride, n))
    for i in range(n):
        for row in range(num_rows):
            for col in range(num_cols):
                l = row * num_cols + col # calculate the index in MX for the current patch
                current_patch = X_ims[row*int(32/num_rows):(row+1)*int(32/num_rows), col*int(32/num_cols):(col+1)*int(32/num_cols), :, i] # extract the patch from input data
                MX[l,:,i] = current_patch.reshape((1, stride*stride*3), order = 'C') # flatten the patch and store it in MX
    return MX

def flatten_filters(Fs):
    f = Fs.shape[0]
    nf = Fs.shape[3]
    flattened_Fs = Fs.reshape((f * f * 3, nf), order='C')
    return flattened_Fs

def convolutional_layer_calculation(MX, flattened_Fs, stride = 4):
    conv_outputs = np.einsum('ijn, jl ->iln', MX, flattened_Fs, optimize=True)
    # num_patches = int(np.sqrt(MX.shape[1]/3) / stride)**2
    # n = MX.shape[2]
    # conv_outputs = np.zeros((num_patches, flattened_Fs.shape[1], n))
    # for i in range(n):
    #     conv_outputs[:, :, i] = np.matmul(MX[:, :, i], flattened_Fs) 
    return conv_outputs
    

def softmax(x):
    z = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

def softmax_s(s): #TODO: check if this is equivalent to the softmax in the instructions
    return np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)

def initializeWeights(k, d, seed=42):
    np.random.seed(seed) 
    weights = np.random.normal(0, 2/np.sqrt(d), (k, d))  
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

def initializeConvModel(filter_dims, num_filters, num_hidden, num_labels, num_patches, seed=42):
    np.random.seed(seed)
    
    f = filter_dims[0]
    
    # Convolutional Layer, single patch (f * f * 3)
    fan_in_conv = f * f * 3
    # Shape: (3*f*f, num_filters)
    filter_vector = np.random.normal(0, np.sqrt(2/fan_in_conv), (fan_in_conv, num_filters))
    filter_bias_vector = np.zeros((num_filters, 1))
    
    # Hidden Layer
    # Fan-in is the total number of activations coming from the flattened conv layer
    fan_in_hidden = num_filters * num_patches
    l1_weights_hidden = np.random.normal(0, np.sqrt(2/fan_in_hidden), (num_hidden, fan_in_hidden))
    l1_bias_hidden = np.zeros((num_hidden, 1))
    
    # Output Layer fan-in is the number of hidden nodes
    fan_in_output = num_hidden
    l2_weights_hidden = np.random.normal(0, np.sqrt(2/fan_in_output), (num_labels, num_hidden))
    l2_bias_hidden = np.zeros((num_labels, 1))

    return {
        "conv_layer": {"weights": filter_vector, "bias": filter_bias_vector},
        "hidden_layer": {"weights": l1_weights_hidden, "bias": l1_bias_hidden},
        "output_layer": {"weights": l2_weights_hidden, "bias": l2_bias_hidden}
    }

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

def computeLoss(a_values, model, labels, l, onehot: bool = False):

    sum1 = np.sum(lcross(labels, a_values[-1], onehot)) 
    
    sum2 = 0
    for layer in model:
        sum2 += np.sum(layer["weights"] ** 2) 

    N = a_values[-1].shape[1] 
    total_loss = (1/N) * sum1 + l * sum2  
    return total_loss 

def ReLU(x):
    return np.maximum(0,x)

def getPredictedLabels(a_values):
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

def BackwardPassConv(MX, conv_flat_activated, x1, p, z1, conv_model, labels, l):
    N = p.shape[1]
    n_p = MX.shape[0]
    nf = conv_model["conv_layer"]["weights"].shape[1] 

    # LAYER L-1 (Output layer with softmax)
    g = p.copy()
    g[labels.flatten(), np.arange(N)] -= 1
    
    dL_dW2 = (1/N) * np.matmul(g, x1.T) + 2 * l * conv_model["output_layer"]["weights"]
    dL_db2 = (1/N) * np.sum(g, axis=1, keepdims=True)
    
    # LAYER L-2 (Hidden layer)
    g = np.matmul(conv_model["output_layer"]["weights"].T, g)
    g = g * (z1 > 0) # ReLU derivative
    
    dL_dW1 = (1/N) * np.matmul(g, conv_flat_activated.T) + 2 * l * conv_model["hidden_layer"]["weights"]
    dL_db1 = (1/N) * np.sum(g, axis=1, keepdims=True)
    
    # LAYER L-3 (Convolutional layer)
    g = np.matmul(conv_model["hidden_layer"]["weights"].T, g)
    
    # G_batch represents the flattened conv output gradient 
    G_batch = g * (conv_flat_activated > 0) # ReLU derivative for conv layer
    GG = G_batch.reshape((n_p, nf, N), order='C')
    MXt = np.transpose(MX, (1, 0, 2))
    
    # Compute gradient for flattened filters using Einsum
    dL_dF_flat = (1/N) * np.einsum('ijn, jln ->il', MXt, GG, optimize=True) 
    dL_dW_conv = dL_dF_flat + 2 * l * conv_model["conv_layer"]["weights"]   #shape: (3*f*f, num_filters)
    dL_db_conv = (1/N) * np.sum(GG, axis=(0, 2)).reshape(-1, 1)
    
    # Return gradients
    return {
        "conv_layer": {"weights": dL_dW_conv, "bias": dL_db_conv},
        "hidden_layer": {"weights": dL_dW1, "bias": dL_db1},
        "output_layer": {"weights": dL_dW2, "bias": dL_db2}
    }

def relativeError(grad1, grad2, eps=1e-8):
    nominator = np.abs(grad1 - grad2)
    denominator = np.maximum(eps, np.abs(grad1) + np.abs(grad2))
    return nominator / denominator


def miniBatchGradientDescentConv(
        MX_train, labels_train_flat, MX_val, labels_val_flat, n_batch, 
         n_epochs, conv_model, lam, n_s = 500, seed=42): 
    
    n = MX_train.shape[2] 
    model_trained = copy.deepcopy(conv_model) 

    train_costs, val_costs = [], []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    update_steps = []
    
    start_time = time.time()

    for epoch in range(n_epochs):
        rng = np.random.RandomState(seed + epoch)
        shuffled_indices = rng.permutation(n) 

        for i in range(n // n_batch):
            i_start = i * n_batch 
            i_end = (i+1) * n_batch 
            inds = np.arange(i_start, i_end) 
            t = epoch * (n // n_batch) + i 

            learningRate = computeCyclicalLearningRate(eta_min=1e-5, eta_max=1e-1, n_s=n_s, t=t)

            # Slice the MX batch (shape is n_p, 3*f*f, n) => slice on the 3rd axis
            mx_batch = MX_train[:, :, shuffled_indices[inds]] 
            y_batch = labels_train_flat[shuffled_indices[inds]] 
            
            # Forward and Backward pass
            conv_flat_activated, x1, p, z1, z2 = conv_forward_pass(mx_batch, model_trained)
            grads = BackwardPassConv(mx_batch, conv_flat_activated, x1, p, z1, model_trained, y_batch, l=lam) 
            
            # Update parameters
            for layer_name in model_trained.keys():
                model_trained[layer_name]["weights"] -= learningRate * grads[layer_name]["weights"]
                model_trained[layer_name]["bias"] -= learningRate * grads[layer_name]["bias"]

        current_step = (epoch + 1) * (n // n_batch)
        update_steps.append(current_step)

        # Evaluate Validation Data at the end of each epoch
        _, _, p_val, _, _ = conv_forward_pass(MX_val, model_trained) 
        
        # adapt computeLoss logic here for the CNN
        val_loss = np.mean(-np.log(np.clip(p_val[labels_val_flat, np.arange(len(labels_val_flat))], 1e-15, 1.0 - 1e-15)))
        val_losses.append(val_loss)
        
        l2_reg = lam * (np.sum(model_trained["conv_layer"]["weights"]**2) + np.sum(model_trained["hidden_layer"]["weights"]**2) + np.sum(model_trained["output_layer"]["weights"]**2))
        val_costs.append(val_loss + l2_reg) 
        
        val_accs.append(computeAccuracy(np.argmax(p_val, axis=0), labels_val_flat))
        
        print(f"Epoch {epoch+1}/{n_epochs} | Val Acc: {val_accs[-1]*100:.2f}% | Val Cost: {val_costs[-1]:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    return model_trained, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps

def miniBatchGradientDescent(
        X_train, labels_train, X_val, labels_val, n_batch, 
         n_epochs, model, lam, n_s = 500, learningRateCalc="static", seed=42): 
    
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
                learningRate = computeCyclicalLearningRate(eta_min=1e-5, eta_max=1e-1, n_s=n_s, t=t)
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
        train_costs.append(computeLoss(a_train, model_trained, labels_train, l=lam)) 
        train_losses.append(computeLoss(a_train, model_trained, labels_train, l=0)) 
        train_accs.append(computeAccuracy(getPredictedLabels(a_train), labels_train))

        # Evaluate Validation Data
        z_val, a_val = applyNetwork(X_val, model_trained) 
        val_costs.append(computeLoss(a_val, model_trained, labels_val, l=lam))  
        val_losses.append(computeLoss(a_val, model_trained, labels_val, l=0)) 
        val_accs.append(computeAccuracy(getPredictedLabels(a_val), labels_val))

    return model_trained, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps 

def conv_forward_pass(MX, conv_model, stride=4):
    n_p = MX.shape[0]
    n = MX.shape[2]
    flattened_Fs = conv_model["conv_layer"]["weights"]

    nf = flattened_Fs.shape[1]
    conv_out = convolutional_layer_calculation(MX, flattened_Fs, stride=stride)
    conv_out += conv_model["conv_layer"]["bias"].reshape((1, nf, 1))
    conv_flat_activated = np.fmax(conv_out.reshape((n_p*nf, n), order='C'), 0)

    z1, x1 = applyLayer(conv_flat_activated, conv_model["hidden_layer"], apply_relu=True)
    
    z2, p = applyLayer(x1, conv_model["output_layer"], apply_relu=False)
    
    return conv_flat_activated, x1, p, z1, z2


def computeCyclicalLearningRate(eta_min, eta_max, n_s, t):
    l = t // (2 * n_s)
    if 2*l*n_s <= t <= (2*l+1)*n_s:
        learningRate = eta_min + ((t - 2*l*n_s) / n_s) * (eta_max - eta_min)
    else:        
        learningRate = eta_max - ((t - (2*l+1)*n_s) / n_s) * (eta_max - eta_min)
    return learningRate

def lambda_search(n_s, dims,train_X, train_y, val_X, val_y,search_range=(-5,-1),seed=42, search_amount=10, n_epochs=8):
    search_results = []
    for i in range(search_amount):
        l_val = np.random.uniform(search_range[0], search_range[1])
        lam = 10 ** l_val

        model = initializeModel(dims, seed=seed+i)
        trained_model, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps = miniBatchGradientDescent(
            train_X, train_y, val_X, val_y, n_batch=100, learningRateCalc="cyclical", n_epochs=n_epochs, model=model, n_s=n_s, lam=lam, seed=seed+i)
        # print("Mini-batch gradient descent completed.")
        best_val_acc = max(val_accs)
        search_results.append((lam, best_val_acc))
        print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")
    
    search_results.sort(key=lambda x: x[1], reverse=True)
    return search_results

 

def main():
    randseed = 42 # np.random.randint(0, 1000) 
    nf_debug = 2
    k_debug = 10
    f_debug = 4
    num_patches_debug = int(32//f_debug)**2



    X_ims_debug, Fs_debug, true_convolutions_debug, true_labels_debug = debug_data_load()
    # print(f'Debug data shapes - X_ims: {X_ims_debug.shape}, Fs: {Fs_debug.shape}, true_labels: {true_labels_debug.shape}')
    print("Debug data loaded successfully.")

    computed_conv_outputs_debug = seq_convolutional_layer_calculation(X_ims_debug, Fs_debug, num_filters=nf_debug, stride=f_debug)
    print("Debug convolutional layer calculation completed.")
    # print(f'Computed convolutional outputs shape: {computed_conv_outputs_debug.shape}')
    # diff = np.abs(computed_conv_outputs_debug - true_labels_debug)

    conv_outputs_flat = computed_conv_outputs_debug.reshape((num_patches_debug, nf_debug, true_convolutions_debug.shape[3]), order='C')
    
    matrix_conv_outputs = convolutional_layer_calculation(MX=MX_initialization(X_ims_debug, stride=f_debug), flattened_Fs=flatten_filters(Fs_debug), stride=f_debug)
    print("Matrix convolutional layer calculation completed.")
    # print(f'Matrix convolutional outputs shape: {matrix_conv_outputs.shape}')
    # print(f'Max absolute difference between sequential and matrix convolutional outputs: {np.max(np.abs(conv_outputs_flat - matrix_conv_outputs))}')

    conv_model = initializeConvModel(filter_dims=(f_debug, f_debug), num_filters=nf_debug, num_hidden=k_debug, num_labels=k_debug, num_patches=num_patches_debug, seed=randseed)
    print("Convolutional model initialized successfully.")

    #============== debug forward pass ==============

    MX = MX_initialization(X_ims_debug, stride=f_debug)
    Fs_flat = flatten_filters(Fs_debug)

    conv_flat_activated, x1, p, z1, z2 = conv_forward_pass(MX,  conv_model, stride=f_debug)
    print("Convolutional forward pass completed.")

    debug_labels_flat = np.argmax(true_labels_debug, axis=0)
    
    grads = BackwardPassConv(MX, conv_flat_activated, x1, p, z1, conv_model, debug_labels_flat, l=0.01)
    print("Convolutional backward pass completed.")

    torch_grads = ComputePytorchGradientsConv(MX, debug_labels_flat, conv_model, lam=0.01)
    print("PyTorch gradient computation completed.")

    print("\n--- Gradient Check (Relative Error) ---")
    for layer_name in grads.keys():
        for param_name in grads[layer_name].keys():
            ag = grads[layer_name][param_name]
            pg = torch_grads[layer_name][param_name]
            
            # Use your relativeError function
            error = relativeError(ag, pg)
            max_error = np.max(error)
            
            print(f"{layer_name} - {param_name}: Max Relative Error = {max_error:.2e}")
            
            if max_error > 1e-5:
                print(f"  --> WARNING: High error detected in {layer_name} {param_name}!")

    
    # ============= end of debug ==============
    total_x, _, total_y = LoadBatch(1)
    
    for batch_num in range(2, 6):
        data, _, labels = LoadBatch(batch_num)
        total_x = np.vstack([total_x, data])
        total_y = np.concatenate([total_y, labels])
        
        
        
    train_X = total_x[:49000, :]
    train_y = total_y[:49000]
    val_X = total_x[49000:, :]
    val_y = total_y[49000:]


    # train_X = total_x[:45000, :]
    # train_y = total_y[:45000]
    # val_X = total_x[45000:, :]
    # val_y = total_y[45000:]
    mean_X, std_X = computeMeanStd(train_X)
    train_X = normalizeData(train_X, mean_X, std_X) 

    val_X = normalizeData(val_X, mean_X, std_X) 


    n_train = train_X.shape[0]
    n_val = val_X.shape[0]

    # Cast to float32
    train_X_ims = np.transpose(train_X.T.reshape((32, 32, 3, n_train), order='F'), (1, 0, 2, 3)).astype(np.float32)
    val_X_ims = np.transpose(val_X.T.reshape((32, 32, 3, n_val), order='F'), (1, 0, 2, 3)).astype(np.float32)

    # create MX matrices
    f_val = 4
    MX_train = MX_initialization(train_X_ims, stride=f_val)
    MX_val = MX_initialization(val_X_ims, stride=f_val)

    # initialize network parameters
    num_filters = 10
    num_hidden = 50
    num_patches = int(32//f_val)**2

    conv_model = initializeConvModel(filter_dims=(f_val, f_val), num_filters=num_filters, num_hidden=num_hidden, num_labels=10, num_patches=num_patches, seed=randseed)

    # train network
    n_batch = 100
    n_s = 800
    n_epochs = int((3 * 2 * n_s) / (n_train // n_batch))
    
    trained_model, _, val_costs, _, val_losses, _, val_accs, update_steps = miniBatchGradientDescentConv(
        MX_train, train_y, MX_val, val_y, 
        n_batch=n_batch, n_epochs=n_epochs, 
        conv_model=conv_model, lam=0.003, n_s=n_s, seed=randseed
    )
    # test_X, _, test_y = LoadBatch(-1)
    # test_X = normalizeData(test_X, mean_X, std_X)
    # print("Test data loaded and normalized.")


    # n_batch = 100
    # n_train = train_X.shape[0] 
    # n_s = 2 * (n_train // n_batch) # Calculate step size dynamically 
    # n_epochs = 12 # n_batch = 100 => 450 update steps, n_s = 2* n/n_batch = 900 => 4 epochs per cycle * 3 cycles = 12 epochs

    
    
    # m = 50 
    # dims = [[m, train_X.shape[1]],
    #         [len(np.unique(train_y)), m]]  

    # # n_epochs_search = 8 # n_batch = 100 => 450 update steps, n_s = 2* n/n_batch = 900 => 4 epochs per cycle * 2 cycles = 8 epochs
    # # search_amount = 10

    # # Coarse search, ran multiple times with differnet seeds to get a good range for the fine search
    # # search_results = lambda_search(n_s, dims, train_X, train_y, val_X, val_y, search_range=(-5,-1), seed=randseed, search_amount=search_amount, n_epochs=n_epochs)
    # # print("\nLambda Search Results (sorted by validation accuracy):")
    # # for lam, val_acc in search_results:
    # #     print(f"Lambda: {lam:.2e}, Best Validation Accuracy: {val_acc * 100:.2f}%")

    # # Fine search 
    # # search_results_fine = lambda_search(n_s, dims, train_X, train_y, val_X, val_y, search_range=(-4.7,-3.5), seed=randseed, search_amount=search_amount+10, n_epochs=n_epochs*2)
    # # print("\nLambda Search Results (sorted by validation accuracy):")
    # # for lam, val_acc in search_results_fine:
    # #     print(f"Lambda: {lam:.2e}, Best Validation Accuracy: {val_acc * 100:.2f}%")





    # model = initializeModel(dims, seed=randseed)
    # trained_model, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, update_steps = miniBatchGradientDescent(
    #         train_X, train_y, val_X, val_y, n_batch=100, learningRateCalc="cyclical", n_epochs=n_epochs, model=model, n_s=n_s, lam=2.64e-04, seed=randseed)
    


    # fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # Create 1 row with 3 columns

    # # Cost Plot
    # axs[0].plot(update_steps, train_costs, label='training cost', color='teal')
    # axs[0].plot(update_steps, val_costs, label='validation cost', color='crimson')
    # axs[0].set_xlabel('update step')
    # axs[0].set_ylabel('cost')
    # axs[0].set_title('Cost plot')
    # axs[0].legend()

    # # Loss Plot
    # axs[1].plot(update_steps, train_losses, label='training loss', color='teal')
    # axs[1].plot(update_steps, val_losses, label='validation loss', color='crimson')
    # axs[1].set_xlabel('update step')
    # axs[1].set_ylabel('loss')
    # axs[1].set_title('Loss plot')
    # axs[1].legend()

    # # Accuracy Plot
    # axs[2].plot(update_steps, train_accs, label='training accuracy', color='teal')
    # axs[2].plot(update_steps, val_accs, label='validation accuracy', color='crimson')
    # axs[2].set_xlabel('update step')
    # axs[2].set_ylabel('accuracy')
    # axs[2].set_title('Accuracy plot')
    # axs[2].legend()

    # plt.tight_layout() # fix layout
    # plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- TORCH-IMPLEMENTATION, USED FOR GRADIENT CHECKING ONLY, FROM DIFFERENT FILE ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def ComputePytorchGradientsConv(MX, labels_flat, conv_model, lam):
#    
#     # 1. Convert input data to PyTorch tensors
#     MX_t = torch.tensor(MX, dtype=torch.float64)
#     targets = torch.tensor(labels_flat, dtype=torch.long)
    
#     # 2. Convert network parameters to PyTorch tensors and require gradients
#     W_conv = torch.tensor(conv_model["conv_layer"]["weights"], dtype=torch.float64, requires_grad=True)
#     b_conv = torch.tensor(conv_model["conv_layer"]["bias"], dtype=torch.float64, requires_grad=True)
    
#     W1 = torch.tensor(conv_model["hidden_layer"]["weights"], dtype=torch.float64, requires_grad=True)
#     b1 = torch.tensor(conv_model["hidden_layer"]["bias"], dtype=torch.float64, requires_grad=True)
    
#     W2 = torch.tensor(conv_model["output_layer"]["weights"], dtype=torch.float64, requires_grad=True)
#     b2 = torch.tensor(conv_model["output_layer"]["bias"], dtype=torch.float64, requires_grad=True)
    
#     n_p = MX.shape[0]
#     n = MX.shape[2]
#     nf = W_conv.shape[1]

#     # FORWARD PASS

#     # Convolutional Layer (Using for-loop)
#     conv_out = torch.zeros((n_p, nf, n), dtype=torch.float64)
#     for i in range(n):
#         conv_out[:, :, i] = torch.matmul(MX_t[:, :, i], W_conv) + b_conv.T
        
#     # Flatten and apply ReLU
#     conv_flat = conv_out.reshape(n_p * nf, n)
#     x1 = torch.clamp(conv_flat, min=0.0)
    
#     # Hidden Layer + ReLU
#     z1 = torch.matmul(W1, x1) + b1
#     x2 = torch.clamp(z1, min=0.0)
    
#     # Output Layer (Logits only, PyTorch handles the Softmax internally)
#     z2 = torch.matmul(W2, x2) + b2
    
#     # LOSS & BACKPROPAGATION
#     # nn.CrossEntropyLoss expects logits of shape (batch_size, num_classes), so we transpose z2
#     criterion = torch.nn.CrossEntropyLoss()
#     loss_ce = criterion(z2.T, targets)
    
#     # L2 Regularization
#     l2_reg = lam * (torch.sum(W_conv**2) + torch.sum(W1**2) + torch.sum(W2**2))
#     total_loss = loss_ce + l2_reg
#     total_loss.backward()
    
#     # Return the gradients
#     return {
#         "conv_layer": {"weights": W_conv.grad.numpy(), "bias": b_conv.grad.numpy()},
#         "hidden_layer": {"weights": W1.grad.numpy(), "bias": b1.grad.numpy()},
#         "output_layer": {"weights": W2.grad.numpy(), "bias": b2.grad.numpy()}
#     }

if __name__ == "__main__":
    main()

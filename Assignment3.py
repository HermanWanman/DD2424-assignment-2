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

    true_labels = load_data['conv_outputs']
    return X_ims, Fs, true_labels

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

    filter_vector = np.random.normal(0, 2/np.sqrt(filter_dims[0] * filter_dims[1] * 3), (num_filters, filter_dims[0] * filter_dims[1] * 3))
    filter_bias_vector = np.zeros((num_filters, 1))
    l1_weights_hidden = initializeWeights(num_hidden, (num_filters*num_patches), seed)
    l1_bias_hidden = initializeBias(num_filters*num_patches)
    l2_weights_hidden = initializeWeights(num_labels, num_hidden, seed)
    l2_bias_hidden = initializeBias(num_labels)

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

def relativeError(grad1, grad2, eps=1e-8):
    nominator = np.abs(grad1 - grad2)
    if np.sum(np.abs(grad1)) + np.sum(np.abs(grad2)) < eps:
        return np.abs(grad1 - grad2)/eps
    else:
        denominator = np.abs(grad1) + np.abs(grad2)
        return nominator/denominator  

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

def conv_forward_pass(MX, flattened_Fs, conv_model, stride=4):
    conv_out = convolutional_layer_calculation(MX, flattened_Fs, stride=stride)
    


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



    X_ims_debug, Fs_debug, true_convolutions_debug = debug_data_load()
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

    #============== Actual forward pass ==============

    conv_forward_pass(MX_initialization(X_ims_debug, stride=f_debug), flatten_filters(Fs_debug), conv_model, stride=f_debug)



    # total_x, _, total_y = LoadBatch(1)
    
    # for batch_num in range(2, 6):
    #     data, _, labels = LoadBatch(batch_num)
    #     total_x = np.vstack([total_x, data])
    #     total_y = np.concatenate([total_y, labels])
        
        
        
    # train_X = total_x[:49000, :]
    # train_y = total_y[:49000]
    # val_X = total_x[49000:, :]
    # val_y = total_y[49000:]


    # # train_X = total_x[:45000, :]
    # # train_y = total_y[:45000]
    # # val_X = total_x[45000:, :]
    # # val_y = total_y[45000:]
    # mean_X, std_X = computeMeanStd(train_X)
    # train_X = normalizeData(train_X, mean_X, std_X) 
    # print("Training data loaded and normalized.")

    # # val_X, _, val_y = LoadBatch(2) 
    # val_X = normalizeData(val_X, mean_X, std_X) 
    # print("Validation data loaded and normalized.")

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
    
    # print("\n--- Final Model Evaluation ---")
    
    # # 1. Run the test images through your fully trained network
    # z_test, a_test = applyNetwork(test_X, trained_model)
    
    # # 2. Get the model's predictions
    # test_predictions = getPredictedLabels(a_test)
    
    # # 3. Calculate the final accuracy against the true test labels
    # test_accuracy = computeAccuracy(test_predictions, test_y)
    
    # # 4. (Optional but good for the report) Calculate the final test cost
    # test_cost = computeLoss(a_test, trained_model, test_y, l=2.64e-04)
    
    # print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    # print(f"Final Test Cost: {test_cost:.4f}")

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


if __name__ == "__main__":
    main()
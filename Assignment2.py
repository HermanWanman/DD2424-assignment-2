import copy

import matplotlib
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch_gradient_computations import ComputeGradsWithTorch
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for better performance
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def LoadBatch(batch_number):
    filename = f'./Datasets/cifar-10-batches-py/data_batch_{batch_number}'
    if batch_number == 5:
        filename = f'./Datasets/cifar-10-batches-py/test_batch'
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes') 
    imagePixelData = data[b'data'].astype(np.float64) / 255.0  # Normalize pixel values to [0, 1]
    imageLabels = np.array(data[b'labels'])  # Convert labels to a numpy array
    oneHotRep = np.zeros((10, len(imageLabels)))  # Create a zero matrix for one-hot encoding
    oneHotRep[imageLabels, np.arange(len(imageLabels))] = 1  # Set the appropriate entries to 1
    print(imagePixelData.shape, oneHotRep.shape, imageLabels.shape)  # Print the shapes of the loaded data
    return imagePixelData, oneHotRep, imageLabels

def softmax(x):
    # x shape: (k, N) where k is number of classes and N is number of samples
    # Stable softmax along the class dimension for each sample
    z = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

def normalizeData(imagePixelData, d):
    # imagePixelData shape: (num_images, d)
    mean_X = np.mean(imagePixelData, axis=0).reshape(1, d)  # Compute the mean of each pixel across all images
    std_X = np.std(imagePixelData, axis=0).reshape(1, d) # Compute the standard deviation of each pixel across all images
    normalized_X = (imagePixelData - mean_X) / std_X  # Normalize the pixel values

    return normalized_X

def initializeWeights(k, d, seed = 42):
    np.random.seed(seed) # Set the random seed according to assignment instructions
    weights = np.random.rand(k, d) * 0.01  # Initialize weights with small random values
    return weights

def initializeBias(k):
    bias = np.zeros((k, 1))  # Initialize bias with zeros
    return bias

def initializeModel(k, d, seed = 42):
    weights = initializeWeights(k, d, seed)  # Initialize weights
    bias = initializeBias(k)  # Initialize bias
    model = {"weights": weights, "bias": bias}  # Create a dictionary to store weights and bias
    return model

def applyNetwork(pixelData, model):
    weights = model["weights"]  # Extract weights from the model
    bias = model["bias"]  # Extract bias from the model
    z = np.matmul(weights, pixelData.T) + bias  # Compute the output of the network using the softmax function per row
    outputProbs = softmax(z)  # Apply softmax to the output of the network

    return outputProbs

def lcross(labels, outputProbs, onehot: bool = False):
    epsilon = 1e-15  # Small constant to prevent log(0)
    safe_outputProbs = outputProbs + epsilon  # Clip output probabilities to prevent log(0)
    if onehot:
        lcross = -labels * np.log(safe_outputProbs)  # Compute the cross-entropy loss using one-hot encoded labels
    else:
        lcross = -np.log(safe_outputProbs[labels, np.arange(labels.shape[0])])  # Compute the cross-entropy loss using integer labels
    return lcross  # Return lcross

def computeLoss(outputProbs, model, labels, l, onehot: bool = False):
    sum1 = np.sum(lcross(labels, outputProbs, onehot))  # Compute the sum of the cross-entropy loss
    sum2 = np.sum(model["weights"] ** 2)  # Compute the regularization term

    N = outputProbs.shape[1]  # Get the number of samples
    total_loss = (1/N) * sum1 + l * sum2  
    return total_loss  # Return the total loss

def getPredictedLabels(outputProbs):
    predicted_labels = np.argmax(outputProbs, axis=0)  # Get the predicted labels by taking the argmax of the output probabilities
    return predicted_labels

def computeAccuracy(predicted_labels, labels):
    accuracy = np.mean(predicted_labels == labels)  # Compute the accuracy by comparing predicted labels with true labels
    return accuracy  # Return the accuracy

def BackwardPass(pixelData, outputProbs, model, labels, l, onehot: bool = False):

    N = pixelData.shape[0]  # Get the number of samples
    if onehot:
        dL_dz = outputProbs - labels  # Compute the gradient of the loss with respect to the output probabilities for one-hot encoded labels
    else:
        dL_dz = outputProbs.copy()  # Copy the output probabilities
        dL_dz[labels, np.arange(N)] -= 1  # Subtract 1 from the appropriate entries for integer labels

    dL_dw = (1/N) * np.matmul(dL_dz, pixelData) + 2 * l * model["weights"]  # Compute the gradient of the loss with respect to the weights
    dL_db = (1/N) * np.sum(dL_dz, axis=1, keepdims=True)  # Compute the gradient of the loss with respect to the bias

    return dL_dw, dL_db  # Return the gradients

def relativeError(grad1, grad2, eps = 1e-8):
    nominator = np.abs(grad1 - grad2)
    if np.sum(np.abs(grad1)) + np.sum(np.abs(grad2)) < eps:
        return np.abs(grad1 - grad2)/eps
    else:
        denominator = np.abs(grad1) + np.abs(grad2)
        return nominator/denominator  

def miniBatchGradientDescent(
        X_train,
        labels_train,
        X_val,
        labels_val,
        n_batch, # number of images per batch
        learningRate, 
        n_epochs, # number of epochs to train the model
        model, # model is a dict with keys "weights" and "bias"
        lam, # regularization parameter
        seed = 42
        ): 
    
    n = X_train.shape[0] # number of training samples
    model_trained = copy.deepcopy(model) # create a copy of the model to update during training

    train_costs = []  # Initialize a list to store the cost
    val_costs = []  # Initialize a list to store the validation cost

    train_losses = []  # Initialize a list to store the training loss
    val_losses = []  # Initialize a list to store the validation loss

    for epoch in range(n_epochs):
        rng = np.random.RandomState(seed + epoch)
        shuffled_indices = rng.permutation(n) # shuffle the indices

        for i in range(n//n_batch):

            i_start = i * n_batch # first elemnt of batch
            i_end = (i+1) * n_batch # last element of batch
            inds = np.arange(i_start, i_end) # indices of the batch
            x_batch = X_train[shuffled_indices[inds], :] # batch of pixel inputdata
            y_batch = labels_train[shuffled_indices[inds]] # batch of labels
            outputprobs_batch = applyNetwork(x_batch, model_trained) # compute output probabilities for the batch

            dL_dw, dL_db = BackwardPass(x_batch, outputprobs_batch, model_trained, y_batch, l=lam, onehot=False) # compute gradients for the batch
            model_trained["weights"] -= learningRate * dL_dw # update weights
            model_trained["bias"] -= learningRate * dL_db # update bias

        train_outputprobs = applyNetwork(X_train, model_trained)  # Apply the network to the entire training data at the end of each epoch

        train_cost = computeLoss(train_outputprobs, model_trained, labels_train, l=lam, onehot=False) # compute cost at the end of each epoch
        train_costs.append(train_cost) 

        train_loss = computeLoss(train_outputprobs, model_trained, labels_train, l=0, onehot=False)  # compute training loss
        train_losses.append(train_loss) 

        val_outputprobs = applyNetwork(X_val, model_trained)  # Apply the network to the entire validation data
        val_cost = computeLoss(val_outputprobs, model_trained, labels_val, l=lam, onehot=False)  # compute validation cost
        val_costs.append(val_cost)  

        val_loss = computeLoss(val_outputprobs, model_trained, labels_val, l=0, onehot=False)  # compute validation loss
        val_losses.append(val_loss)

    return model_trained, train_costs, val_costs, train_losses, val_losses # return the updated model after training



def main():

    train_X, _, train_y = LoadBatch(1) # Load training data from the cifar-10 dataset
    train_X = normalizeData(train_X, train_X.shape[1]) # Normalize the training data
    print("Training data loaded and normalized.")

    val_X, _, val_y = LoadBatch(2) 
    val_X = normalizeData(val_X, val_X.shape[1]) 
    print("Validation data loaded and normalized.")

    test_X, _, test_y = LoadBatch(5)
    test_X = normalizeData(test_X, test_X.shape[1])
    print("Test data loaded and normalized.")

    model = initializeModel(len(np.unique(train_y)), train_X.shape[1]) # Initialize the model as a dict
    print("Model initialized with weights and bias.")

    output = applyNetwork(train_X, model)  # Apply the network to the training data
    print("Network applied to training data.")

    cross_entropy_loss = computeLoss(output, model, train_y, l=0.01, onehot=False)  # Compute the cross-entropy loss
    print(f"Cross-entropy loss computed: {cross_entropy_loss}")

    get_predicted_labels = getPredictedLabels(output)  # Get the predicted labels from the output probabilities
    print("Predicted labels obtained from output probabilities.")

    accuracy = computeAccuracy(get_predicted_labels, train_y)  # Compute the accuracy of the predictions
    print(f"Accuracy computed: {accuracy}")

    dL_dw, dL_db = BackwardPass(train_X, output, model, train_y, l=0.01, onehot=False)  # Compute the gradients
    print("Gradients computed.")

    _, torch_gradients = ComputeGradsWithTorch(train_X, train_y, model)  # Compute gradients using PyTorch for comparison
    tgW = torch_gradients['weights']  # Extract the gradient of the weights from PyTorch
    tgb = torch_gradients['bias']  # Extract the gradient of the bias from PyTorch

    # Absolute difference between my gradients and the torch gradients
    abs_diff_W = np.abs(dL_dw - tgW)  
    abs_diff_b = np.abs(dL_db - tgb)  
    print(f'Absolute difference in gradients for weights: {abs_diff_W}')
    print(f'Absolute difference in gradients for bias: {abs_diff_b}')

    # Relative error between my gradients and the torch gradients
    rel_error_W = relativeError(dL_dw, tgW)
    rel_error_b = relativeError(dL_db, tgb)
    print(f'Relative error in gradients for weights: {rel_error_W}')
    print(f'Relative error in gradients for bias: {rel_error_b}')

    trained_model, train_costs, val_costs, train_losses, val_losses = miniBatchGradientDescent(
        X_train=train_X,
        labels_train=train_y,
        X_val=val_X,
        labels_val=val_y,
        n_batch=100,
        learningRate=0.1,
        n_epochs=40,
        model=model,
        lam=0,
        seed=42)

    # Plot the cost over epochs
    plt.plot(train_costs, label='Training Cost')
    plt.plot(val_costs, label='Validation Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost over Epochs')
    plt.legend()
    plt.show()

    # Plot the loss over epochs
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

    Ws = trained_model['weights'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    for i in range(10):
        plt.figure()
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        plt.imshow(w_im_norm)
        plt.title(f'Visualization of Weights for Class {i}')
        plt.savefig(f'./images/class_{i}.png')
        plt.close()

    

    final_output_train = applyNetwork(train_X, trained_model)
    final_predicted_labels_train = getPredictedLabels(final_output_train)
    final_accuracy_train = computeAccuracy(final_predicted_labels_train, train_y)
    print(f"Final accuracy after training: {final_accuracy_train}")

    final_output_validation = applyNetwork(val_X, trained_model)
    final_predicted_labels_validation = getPredictedLabels(final_output_validation)
    final_accuracy_validation = computeAccuracy(final_predicted_labels_validation, val_y)
    print(f"Final accuracy on validation set: {final_accuracy_validation}")

    final_output_test = applyNetwork(test_X, trained_model)
    final_predicted_labels_test = getPredictedLabels(final_output_test)
    final_accuracy_test = computeAccuracy(final_predicted_labels_test, test_y)
    print(f"Final accuracy on test set: {final_accuracy_test}")


if __name__ == "__main__":
    main()
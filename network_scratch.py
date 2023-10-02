import time
import numpy as np
import utils



class NeuralNetwork():
    def __init__(self, layer_shapes, epochs=50, learning_rate=0.01, random_state=1):
        
        #Define learning paradigms
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        #Define network architecture: no. of layers and neurons
        #layer_shapes[i] is the shape of the input that gets multiplied 
        #to the weights for the layer (e.g. layer_shapes[0] is 
        #the number of input features)
        
        self.layer_shapes = layer_shapes
        self.weights = self._initialize_weights()
        
        #Initialize weight vectors calling the function
        #Initialize list of layer inputs before and after  
        #activation as lists of zeros.
        self.A = [None] * len(layer_shapes)
        self.Z = [None] * (len(layer_shapes)-1)

        #Define activation functions for the different layers
        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv



    def _initialize_weights(self):

        np.random.seed(self.random_state)
        self.weights = [] 

        for i in range(1, len(self.layer_shapes)):
            weight = np.random.rand(self.layer_shapes[i], self.layer_shapes[i-1]) - 0.5
            self.weights.append(weight)

        return self.weights



    def _forward_pass(self, x_train):
        # Setting the first input as the input data
        self.A[0] = x_train
        # Loop through each layer of the network until the second last one
        for i in range(0, len(self.weights) - 1):
            # Calculate the weighted sum of inputs for the current layer
            self.Z[i] = np.dot(self.weights[i], self.A[i])
            # Apply the activation function to get the output for the current layer 
            self.A[i + 1] = self.activation_func(self.Z[i])
        # Compute the weighted sum for the last layer
        self.Z[-1] = np.dot(self.weights[-1], self.A[-2])
        # Apply the output function to the final layer to get a probability distribution over classes
        self.A[-1] = self.output_func(self.Z[-1])
        return self.A[-1]



    # Disclaimer: I got inspired by ChatGPT in this TODO question and then completed the codes
    def _backward_pass(self, y_train, output):
        # Initialize the deltas (errors) for each layer
        deltas = [None] * len(self.weights)
        # Compute the delta for the output layer based on the difference between predicted and actual values
        deltas[-1] = self.cost_func_deriv(y_train, output) * self.output_func_deriv(self.Z[-1])
        # Then compute the deltas for the remaining layers
        for i in range(len(deltas) -2, -1, -1):
            # Delta is calculated based on the error of the next layer and the derivative of the activation function at the current layer
            deltas[i] = np.dot(self.weights[i + 1].T, deltas[i + 1]) * self.activation_func_deriv(self.Z[i])
        # Compute the gradient of the weights for each layer based on deltas and layer outputs
        weight_gradients = []
        for i in range(len(deltas)):
            weight_gradients.append(np.outer(deltas[i], self.A[i]))
        # Return the list of weight gradients for each layer
        return weight_gradients



    def _update_weights(self,weight_gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]



    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )
        
        return train_accuracy, val_accuracy



    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)



    def predict(self, x):
        output = self._forward_pass(x)
        return np.argmax(output)



    def fit(self, x_train, y_train, x_val, y_val):

        history = {'accuracy': [], 'val_accuracy': []}
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self._forward_pass(x)
                weight_gradients = self._backward_pass(y, output)
                self._update_weights(weight_gradients)

            train_accuracy, val_accuracy = self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
        return history

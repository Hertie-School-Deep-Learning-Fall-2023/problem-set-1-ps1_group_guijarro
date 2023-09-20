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
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        pass



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a list of the weight gradients which are used to update the weights in self._update_weights().

        '''
        pass 
    


    def _update_weights(self,weight_gradients):
        '''
        TODO: Update the network weights according to stochastic gradient descent.

        '''
        pass



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
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        pass



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

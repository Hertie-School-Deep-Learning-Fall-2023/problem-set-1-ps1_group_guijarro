import time
import torch
import torch.nn as nn
import torch.optim as optim



class NeuralNetworkTorch(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):

        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state   
        torch.manual_seed(self.random_state)
        
        # Disclaimer: I got inspired by ChatGPT in this TODO question and then completed the codes
        # Initialize an empty list to store layers of the neural network
        layers = []
        # Loop over all but the last size in the 'sizes' list. We subtract 2 here because we are looking at pairs of sizes (current and next)
        for i in range(len(sizes) - 2):
            # Add a Sigmoid activation function after each linear layer except for the last one
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.Sigmoid())
        # After adding all but the last linear layer with their activations, then we add the final linear layer without any activation following it
        # This layer connects the second last size (`sizes[-2]`) to the last size (`sizes[-1]`)
        layers.append(nn.Linear(sizes[-2],sizes[-1]))
        # Convert our list of layers into a PyTorch Sequential model. Using (*) to unpack the list items as arguments
        self.network = nn.Sequential(*layers)

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)


        
    def _forward_pass(self, x_train):
        return self.network(x_train)



    def _backward_pass(self, y_train, output):
        # Compute the loss and backpropagate the error
        loss = self.loss_func(output, y_train.float())
        loss.backward()



    def _update_weights(self):
        # Update the network weights based on the gradient of the loss function
        self.optimizer.step()



    def _flatten(self, x):
        return x.view(x.size(0), -1)       



    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )
        return train_accuracy, val_accuracy



    def predict(self, x):
        # Generate predictions from the network logits and determine the class label
        with torch.no_grad():
            x = self._flatten(x)
            logits = self._forward_pass(x)
            probs = torch.sigmoid(logits)
            return torch.argmax(probs, dim=1)



    def fit(self, train_loader, val_loader):
        start_time = time.time()
        history = {'accuracy': [], 'val_accuracy': []} 

        for iteration in range(self.epochs): 
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x) 
                self._backward_pass(y, output)
                self._update_weights()

            train_accuracy, val_accuracy = self._print_learning_progress(start_time, iteration, train_loader, val_loader)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

        return history



    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)

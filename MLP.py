import numpy as np


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialisation des poids et des biais
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            bias = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'weight': weight, 'bias': bias})

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        activations = [x]
        input = x

        for layer in self.layers:
            z = np.dot(input, layer['weight']) + layer['bias']
            input = self.sigmoid(z)
            activations.append(input)

        return activations

    def backward(self, activations, y_true, learning_rate):
        # Calcul de l'erreur à la sortie
        error = activations[-1] - y_true
        delta = error * self.sigmoid_derivative(activations[-1])

        # Mise à jour des poids et des biais
        for i in reversed(range(len(self.layers))):
            a = activations[i]
            weight_update = np.dot(a.T, delta)
            self.layers[i]['weight'] -= learning_rate * weight_update
            self.layers[i]['bias'] -= learning_rate * \
                delta.mean(axis=0, keepdims=True)

            if i != 0:
                delta = np.dot(
                    delta, self.layers[i]['weight'].T) * self.sigmoid_derivative(activations[i])

    def train(self, x_train, y_train, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            activations = self.forward(x_train)
            self.backward(activations, y_train, learning_rate)

            if epoch % 20 == 0:
                loss = np.mean((activations[-1] - y_train) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")
                losses.append(loss)
        return losses

    def predict(self, x):
        activations = self.forward(x)
        return activations[-1]

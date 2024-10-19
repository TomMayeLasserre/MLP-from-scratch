
# Multilayer Perceptron (MLP)

### Definition:

An MLP is a type of feedforward neural network consisting of multiple layers of neurons, where each layer is fully connected to the next. Given an input vector \( \mathbf{x} \in \mathbb{R}^n \), the output of the MLP can be described mathematically as follows:

### Input Layer:

The input to the MLP is the vector \( \mathbf{x} = [x_1, x_2, \dots, x_n] \). There is no computation done at this layer.

### Hidden Layer(s):

For a hidden layer \( h \), the output \( \mathbf{h}^{(l)} \) is computed as:
\[\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})\]

Where:
- \( \mathbf{h}^{(l-1)} \) is the input to the layer (or the output of the previous layer).
- \( \mathbf{W}^{(l)} \in \mathbb{R}^{m_l 	imes m_{l-1}} \) is the weight matrix.
- \( \mathbf{b}^{(l)} \in \mathbb{R}^{m_l} \) is the bias vector.
- \( \sigma \) is an activation function, typically \( \sigma(x) = rac{1}{1 + e^{-x}} \) (sigmoid) or \( \sigma(x) = \max(0, x) \) (ReLU).

### Output Layer:

The final layer computes the output as:
\[\mathbf{y} = \sigma(\mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)})\]

Where:
- \( \mathbf{W}^{(L)} \in \mathbb{R}^{m_L 	imes m_{L-1}} \) is the weight matrix for the output layer.
- \( \mathbf{b}^{(L)} \in \mathbb{R}^{m_L} \) is the bias for the output layer.
- \( \mathbf{y} \) represents the final output of the MLP.

### Backpropagation:

The weight updates during training are governed by the backpropagation algorithm. The gradients of the loss function \( \mathcal{L}(\mathbf{y}, \mathbf{t}) \) (where \( \mathbf{t} \) is the true label) with respect to the weights are given by:

\[rac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} \mathbf{h}^{(l-1)^T}\]

Where:
- \( \delta^{(l)} = rac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} \cdot \sigma'(\mathbf{h}^{(l)}) \) is the error term at layer \( l \).
- \( \sigma'(x) \) is the derivative of the activation function.

import data
import numpy as np


def softmax(inp):
    exp_scores = np.exp(inp)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def tanh(X):
    return np.tanh(X)

def tanh_deriv(X):
    # tanh deriv: 1 - tanh^2,
    return 1 - np.power(np.tanh(X), 2)

def identity(X):
    return X

def identity_deriv(X):
    return np.ones(X.shape)


func_map = {
    tanh: tanh_deriv,
    identity: identity_deriv,
}


class Layer():
    def __init__(self, in_size, out_size, activation_func):
        # initialize random params (weights and bias)
        self.weights = np.random.randn(in_size, out_size) / np.sqrt(in_size)
        self.bias = np.zeros((1, out_size))
        self.activation_func = activation_func
        self.activation_deriv = func_map[activation_func]

        self.in_size = in_size
        self.out_size = out_size
        self.name = activation_func.__name__

    def __call__(self, inputs):
        # cache last inputs for backprop
        self.last_inputs = inputs

        # compute net input to the activation func
        # which uses the output of the previous layer,
        # then compute this layer's output through the activation func
        net = np.dot(inputs, self.weights) + self.bias
        self.last_net = net
        return self.activation_func(net)

    def __repr__(self):
        return 'Layer(in={},out={},activation={})'.format(self.in_size,
                                                          self.out_size,
                                                          self.name)


class NeuralNet():
    def add(self, layer):
        self.layers.append(layer)

    def __init__(self):
        self.layers = []

    def forward(self, X):
        # forward propagation
        outs = [X]
        for layer in self.layers:
            outs.append(layer(outs[-1]))

        # `outs` are the outputs of each layer
        return outs

    def train(self, X, y, epochs=20000, eta=0.01, lmbda=0.01, verbose=True):
        """
        - eta = learning rate
        - lmbda = regularization param
        """
        n = len(X)
        for i in range(epochs):
            outs = self.forward(X)

            # backpropagation
            # err = outs[-1]
            # err[range(n), y] -= 1

            # for regression:
            err = 2. * (y[:,None] - outs[-1]) / n

            dws = []
            dbs = []
            for l, out in zip(reversed(self.layers), reversed(outs[:-1])):
                dw = (out.T).dot(err)
                db = np.sum(err, axis=0, keepdims=True)

                err = err.dot(l.weights.T) * l.activation_deriv(l.last_inputs)

                # incorporate regularization term
                dw += lmbda * l.weights

                # update the params at the end
                dws.append(dw)
                dbs.append(db)

            # update the params
            for l, dw, db in zip(self.layers, reversed(dws), reversed(dbs)):
                l.weights += -eta * dw
                l.bias += -eta * db

            if verbose and i % 1000 == 0:
                print('Loss: {:.2f}'.format(self.loss(X, y, lmbda)))

    def predict(self, X):
        outs = self.forward(X)

        # final output
        final = outs[-1]

        # return the most likely class
        return np.argmax(final, axis=1)

    def loss(self, X, y, lmbda):
        n = len(X)
        outs = self.forward(X)
        probs = outs[-1]

        # cross-entropy loss
        # logprobs = -np.log(probs[range(n), y])
        # the_loss = np.sum(logprobs)

        # MSE
        the_loss = np.square(np.sum(y[:,np.newaxis] - outs[-1]))

        # regularization term
        the_loss += lmbda/2 * sum(np.sum(np.square(l.weights)) for l in self.layers)

        return 1/n * the_loss


def check_gradient():
    eps = 0.0001
    params = np.array([1,1,1])

    # numerical gradient
    num_grad = (cost(params + eps) - cost(params - eps)) / (2 * eps)



if __name__ == '__main__':

    # Classification
    # X, y = data.make_moons(n_samples=200, noise_std=0.1)
    # nn = NeuralNet()
    # nn.add(Layer(2, 3, tanh))
    # nn.add(Layer(3, 2, softmax))
    # nn.train(X, y, verbose=True)

    # Regression
    n_dimensions = 3
    X, y, theta = data.make_linear(n_samples=200, n_dimensions=n_dimensions, noise_std=0.1)
    nn = NeuralNet()
    nn.add(Layer(n_dimensions, 3, tanh))
    nn.add(Layer(3, 1, identity))
    nn.train(X, y, verbose=True)

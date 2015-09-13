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


class Layer():
    def __init__(self, in_size, out_size, activation_func):
        # initialize random params (weights and bias)
        self.weights = np.random.randn(in_size, out_size) / np.sqrt(in_size)
        self.bias = np.zeros((1, out_size))
        self.activation_func = activation_func

    def __call__(self, inputs):
        # cache last inputs for backprop
        self.last_inputs = inputs

        # compute net input to the activation func
        # which uses the output of the previous layer,
        # then compute this layer's output through the activation func
        net = np.dot(inputs, self.weights) + self.bias
        return self.activation_func(net)


class NeuralNet():
    def __init__(self, in_dim=2, hidden_dim=3, out_dim=2):
        layer_sizes = [in_dim, hidden_dim]

        # initialize layers
        self.layers = []
        for in_size, next_size in zip(layer_sizes, layer_sizes[1:]):
            self.layers.append(Layer(in_size, next_size, tanh))

        output_layer = Layer(hidden_dim, out_dim, softmax)
        self.layers.append(output_layer)

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
            err = outs[-1]
            err[range(n), y] -= 1

            # for regression:
            # err = y - outs[-1]

            dws = []
            dbs = []
            for l, out in zip(reversed(self.layers), reversed(outs[:-1])):
                dw = (out.T).dot(err)
                db = np.sum(err, axis=0, keepdims=True)

                err = err.dot(l.weights.T) * tanh_deriv(l.last_inputs)

                # incorporate regularization term
                dw += lmbda * l.weights

                # update the params at the end
                dws.append(dw)
                dbs.append(db)

            # update the params
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
        logprobs = -np.log(probs[range(n), y])
        the_loss = np.sum(logprobs)

        # regularization term
        the_loss += lmbda/2 * sum(np.sum(np.square(l.weights)) for l in self.layers)

        return 1/n * the_loss


def true_function(X):
    """
    Computes true outputs (from randomly generated parameters)
    """
    # Create random parameters for X's dimensions, plus one for x_0.
    true_theta = np.random.rand(X.shape[1] + 1)
    return true_theta[0] + np.dot(true_theta[1:], X.T)


if __name__ == '__main__':
    from sklearn import datasets
    X, y = datasets.make_moons(n_samples=200, noise=0.2)

    nn = NeuralNet(in_dim=2, hidden_dim=3, out_dim=2)
    nn.train(X, y, verbose=True)


# TO DO implement for regression
# if __name__ == '__main__':
    # n_samples = 1000
    # n_dimensions = 3
    # iterations = 100000

    # print('Samples:', n_samples)
    # print('Dimensions:', n_dimensions)
    # print('Iterations:', iterations)

    # # Prep sample data
    # X = np.random.rand(n_samples, n_dimensions)
    # y = true_function(X)
    # y = np.array([y]).T

    # nn = NeuralNet(in_dim=3, hidden_dim=3, out_dim=3)
    # nn.train(X, y, verbose=True)

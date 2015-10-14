import data
import numpy as np


class Cost():
    def __call__(self, y, y_hat): raise NotImplementedError
    def deriv(self, y, y_hat): raise NotImplementedError

class MeanSquaredError(Cost):
    def __call__(self, y, y_hat):
        sq_errors = np.square(y_hat - y)
        return np.mean(sq_errors)

    def deriv(self, y, y_hat):
        return 2 * np.mean(y_hat - y)

class CrossEntropyLoss(Cost):
    # aka "log loss", "logistic loss"
    # y_hat should be the probabilities of each class,
    # not the class labels

    def __call__(self, y, y_hat):
        return -np.mean(y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

    def deriv(self, y, y_hat):
        return np.mean(y_hat - y)


class Activation():
    def __call__(self, X): raise NotImplementedError
    def deriv(self, X): raise NotImplementedError

class Softmax(Activation):
    def __call__(self, X):
        probs = np.exp(X)/np.sum(np.exp(X))
        return probs

    def deriv(self, X):
        return self(X) * (1 - self(X))

class Tanh(Activation):
    def __call__(self, X):
        return np.tanh(X)

    def deriv(self, X):
        # tanh deriv: 1 - tanh^2,
        return 1 - np.power(np.tanh(X), 2)

class Identity(Activation):
    def __call__(self, X):
        return X

    def deriv(self, X):
        return np.ones(X.shape)


class Layer():
    def __init__(self, in_size, out_size, activation_func):
        if not isinstance(activation_func, Activation):
            raise Exception("Activation func must be an Activation instance")

        # initialize random params (weights and bias)
        self.weights = np.random.randn(in_size, out_size) / np.sqrt(in_size)
        self.bias = np.zeros((1, out_size))
        self.activation = activation_func

        self.in_size = in_size
        self.out_size = out_size
        self.name = self.activation.__class__.__name__

    def net(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def __repr__(self):
        return 'Layer(in={},out={},activation={})'.format(self.in_size,
                                                          self.out_size,
                                                          self.name)


class NeuralNet():
    def add(self, layer):
        self.layers.append(layer)

    def __init__(self, cost_func):
        if not isinstance(cost_func, Cost):
            raise Exception("Cost func must be an Cost instance")

        self.layers = []
        self.cost = cost_func

    def train(self, X, y, epochs=20000, eta=0.01, lmbda=0.01):
        """
        - eta = learning rate
        - lmbda = regularization param
        """

        for epoch in range(epochs):
            # forward propagation
            outs = [X]
            nets = []
            for layer in self.layers:
                net = layer.net(outs[-1])
                out = layer.activation(net)

                nets.append(net)
                outs.append(out)

            # backpropagation
            output_layer = self.layers[-1]
            print(y)
            print(outs[-1])
            print('cost deriv', self.cost.deriv(y, outs[-1]))
            print('np grad', np.gradient(self.cost.deriv, ))
            err = self.cost.deriv(y, outs[-1]) * output_layer.activation.deriv(nets[-1])
            print('OUTPUT ERR', err)

            # deltas for weights and biases
            dws = []
            dbs = []

            for i in reversed(range(len(self.layers))):
                l = self.layers[i]
                dw = (outs[i-1].T).dot(err)
                db = np.sum(err)

                err = err.dot(l.weights.T) * l.activation.deriv(l.last_net)

                # TODO regularization term
                # incorporate regularization term
                # dw += lmbda * l.weights

                # update the params at the end
                dws.append(dw)
                dbs.append(db)

            # update the params
            for l, dw, db in zip(self.layers, reversed(dws), reversed(dbs)):
                # TODO eta
                # l.weights += -eta * dw
                # l.bias += -eta * db
                l.weights += dw
                l.bias += db

            if epoch % 1000 == 0:
                # TODO regularization
                # print('Loss: {:.2f}'.format(self.cost(X, y, lmbda)))
                print('Loss: {:.2f}'.format(self.cost(y, outs[-1])))

            return


if __name__ == '__main__':

    # Regression
    n_dimensions = 3
    X, y, theta = data.make_linear(n_samples=10, n_dimensions=n_dimensions, noise_std=0.1)
    nn = NeuralNet(MeanSquaredError())
    nn.add(Layer(n_dimensions, 3, Tanh()))
    nn.add(Layer(3, 1, Identity()))
    nn.train(X, y)

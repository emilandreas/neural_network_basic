import linearNeuron
import tanhNeuron
import sigmoidNeuron
import reluNeuron
import numpy as np
import math as math

from cost_functions import *
from gradientDescent import *


class NeuralNetwork:
    def __init__(self, w_i, w_h, n_h, w_o,
                 hidden_neuron_type=reluNeuron,
                 output_neuron_type=linearNeuron):
        self.count = 0 #debug
        # Architecture Parameters:
        self.input_width  = w_i
        self.hidden_width = w_h
        self.hidden_depth = n_h
        self.output_width = w_o
        self.n_layer      = self.hidden_depth + 1

        # Non-Linearity of the individual network layer:
        # -) Sigmoid Neuron
        # -) Tanh Neuron
        # -) ReLu Neuron
        # -) Linear Neuron
        self.g_h        = hidden_neuron_type.g # Hidden unit neuron type
        self.g_o        = output_neuron_type.g # Output unit neuron type
        self.grad_g_h   = hidden_neuron_type.grad_g # Gradient of hidden neurons
        self.grad_g_o   = output_neuron_type.grad_g # Gradient of output neurons

        # Weight Initialization:
        # The weights should be initialize to a small non-zero value and the initial weights need to be asymmetric.
        # Typically the weights are i.i.d. and drawn from a normal or uniform distribution, where the variance is
        # scaled w.r.t. to in- (n) and output (m) connections.
        # -) Uniform distribution   i.e. ~ U(-sqrt(6/(n+m)), +sqrt(6/(n+m)))
        # -) Normal distribution 1  i.e. ~ sqrt(2/n) N(0, 1)
        # -) Normal distribution 2  i.e. ~ sqrt(2/(n+m)) N(0, 1)
        self.w_uniform  = lambda n,m:np.random.uniform(-math.sqrt(6.0/(n+m)), math.sqrt(6.0/(m+n) )) # Fill in Lambda Function
        self.w_normal_1 = lambda n: math.sqrt(2.0/n) * np.random.normal(0, 1)# Fill in Lambda Function
        self.w_normal_2 = lambda n, m: math.sqrt(2.0/(n+n)) * np.random.normal(0,1)# Fill in Lambda Function
        self.weight_pdf = self.w_normal_1

        # The biases can be initialized to 0 or to a small positive number (0.001) to make the ReLu Units active for
        # the input distribution
        self.b_0        = 0.001# Fill in Value

        # Regularization
        # -) no normalization        i.e. J_reg(w) = 0
        # -) l1 norm regularization  i.e. J_reg(w) = alpha * ||w||_1
        # -) l2 norm regularization  i.e. J_reg(w) = alpha * ||w||_2
        #self.reg =          # Optional Function Handle
        #self.grad_reg =     # Optional Function Handle
        #self.reg_alpha =    # Optional Constant

        # Initialize the cost function:
        # Parametrization of the cost function and an additive regularization term. Currently, the MSE cost function
        # and l1-, l2-norm can be used for regularization.
        self.j_entropy      = j_mse # Entropy cost function
        self.grad_j_entropy = grad_j_mse# Gradient of the Mean Squared Error
        self.j_reg          = l2_norm # Additive regularization cost function
        self.grad_j_reg     = grad_l2_norm# Gradient of the additive regularization cost function

        # Training of the Neural Network:
        # The neural network is trained via stochastic gradient descent with varying batch sizes and early stopping to
        # prevent overfitting.
        self.sgd = {'j min': 1e-6,
                    'max epoch': 1000,
                    'batch size': 10,
                    'alpha': 1e-4,
                    'p train': 0.8,
                    'p test': 0.1,
                    'p val': 0.1}

        # -) Momentum:
        # To increase convergence speed, momentum accumulates an exponential mean of the previous gradients and adds
        # this accumulated gradient to the current gradient.
        self.momentum = {'select': False,
                         'rho': 0.9}

        # -) Adaptive Learning Rates:
        # To prevent overshooting, the learning rate is decreased by the accumulated squared gradient.
        self.rmsprop = {'select': False,
                        'rho': 0.9}
        # -) ADAM:
        # Combines the concept of momentum with the adaptive learning rate to increase convergence speed and prevent
        # overshooting.
        self.adam = {'select': True,
                     'alpha': 1e-3,
                     'beta 1': 0.95,
                     'beta 2': 0.9}

        # State Variables:
        self.x_train    = None
        self.y_train    = None
        self.w          = np.full((self.n_layer, self.hidden_width, self.hidden_width), 0, np.float)   # Weights of the respective layer
        self.b          = np.full((self.n_layer, self.hidden_width), 0, np.float)   # Bias of the respective layer
        self._grad_w    = []   # Weight change of the respective layer
        self._grad_b    = []   # Bias change of the respective layer
        self._out_i       = np.full((self.n_layer, self.hidden_width), 0, np.float)   # Activation of the respective layer
        self._net_i     = np.full((self.n_layer, self.hidden_width), 0, np.float)   # net values
        self._g_i       = [0]*self.n_layer   # Non-Linearity of the respective layer
        self._grad_g_i  = [0]*self.n_layer   # Gradient of the non-linearity of the respective layer
        self._delta     = np.full((self.n_layer, self.hidden_width), 0, np.float)   # The deltas of each layer
        self._optimizer = None
        self._trained   = False


        for i in range(0, self.n_layer-1):
            self._g_i[i] = np.vectorize(self.g_h)
            self._grad_g_i[i] = np.vectorize(self.grad_g_h)
        self._g_i[self.n_layer-1] = np.vectorize(self.g_o)
        self._grad_g_i[self.n_layer-1] = np.vectorize(self.grad_g_o)

        # Debugging Variables:
        self._fig = None
        self._fig_j = None

        # Initialize the weight matrix:
        #
        #
        # Fill in
        
        self.weight_initialization()



    def eval(self, x):
        # Evaluate the network output

        activ_func_hidden = np.vectorize(self.g_h)
        activ_func_output = np.vectorize(self.g_o)
        #Input layer
        temp_h_i = np.dot(x, self.w[0][ :self.input_width])
        temp_h_i += self.b[0]
        self._net_i[0] = temp_h_i
        self._out_i[0] = activ_func_hidden(temp_h_i)
        #hiddn layers
        for layer in range(1, self.hidden_depth):
            temp_h_i = np.dot(self._out_i[layer -1], self.w[layer, :self.hidden_width,:])
            temp_h_i += self.b[layer]
            self._net_i[layer] = temp_h_i
            self._out_i[layer] = activ_func_hidden(temp_h_i)
        #output
        temp_h_i = np.dot(  self._out_i[self.n_layer-2], self.w[self.hidden_depth, :self.hidden_width, :])
        temp_h_i += self.b[self.n_layer-1]
        self._net_i[self.n_layer-1] = temp_h_i
        self._out_i[self.n_layer-1] = activ_func_output(temp_h_i)
        retVal = self._out_i[self.n_layer-1]
        return self._out_i[self.n_layer-1]

    def weight_initialization(self):
        # Initialize all parameters
        #               w[layer][output][input]

        for a in range(1, 10):
            for b in range(1, 10):
                print "\t", a, " ", b, ": ", self.w_uniform(a, b)
        #Init weights
        #Input
        for o in range(0, self.input_width):
            for i in range(0, self.hidden_width):
                self.w[0][o][i] = self.w_uniform(self.input_width, self.hidden_width)
        #Hidden layers
        for l in range(1,self.hidden_depth):
            for o in range(0,self.hidden_width):
                for i in range(0,self.hidden_width):
                    self.w[l][o][i] = self.w_uniform(self.hidden_width, self.hidden_width)
        #Output
        for o in range(0, self.hidden_width):
            for i in range(0,self.output_width):
                self.w[self.hidden_depth][o][i] = self.w_uniform(self.input_width, self.output_width)

        #Init biases
        #Hidden layers
        for l in range(0, self.hidden_depth):
            for w in range(0, self.hidden_width):
                self.b[l][w] = self.b_0
        #Output
        for w in range(0, self.output_width):
            self.b[self.n_layer-1][w] = self.b_0
        return 0

    def printWeights(self):
        #inputs
        print "Weights: \n"
        print "Layer Input\n"
        for o in range(0, self.input_width):
            for i in range(0, self.hidden_width):
                print "\tnode " , o , ", " , i , ": " , self.w[0][o][i] , "\n"

        #Hidden layers
        print "Layer Hidden \n"
        for l in range(1,self.hidden_depth):
            print "Layer " , l , "\n"
            for o in range(0,self.hidden_width):
                for i in range(0,self.hidden_width):
                    print "\tnode " , o , ", " , i , ": " , self.w[l][o][i] , "\n"
        #Output
        print "Layer Output\n"
        for o in range(0, self.hidden_width):
            for i in range(0,self.output_width):
                print "\tnode " , o , ", " , i , ": " , self.w[self.hidden_depth][o][i] , "\n"
        
        print "Biases: \n"
        for l in range(0, self.hidden_depth):
            print "Layer ", l, "\n"
            for w in range(0, self.hidden_width):
                print "\tnode " , w , ": " , self.b[l][w] , "\n"
        #Output
        print "Layer output\n"
        for w in range(0, self.output_width):
            print "\tnode output" , w , ": " , self.b[self.n_layer-1][w] , "\n"
        return 0

    def j(self, x, y):
        # Network Cost Function
        y_hat = self.eval(x)
        return self.j_entropy(y, y_hat)

    def grad_j(self, x_in, y_in):
        # Gradient w.r.t. to all parameters
        self.count = self.count + 1
        c = self.count
        a = x_in
        b = y_in
        assert y_in.shape[0] == x_in.shape[0]
        grad_E = self.w * 0;
        (rows, columns) = x_in.shape

        #For each input/output in batch
        for itr in range(0, rows):
            x = x_in[itr]
            y = y_in[itr]
            y_hat = self.eval(x)[:self.output_width]
            for layer in reversed(range(0, self.n_layer)):
                #Output layer
                if layer == self.n_layer-1:
                    net = self._net_i[layer, :self.output_width]
                    self._delta[layer, :self.output_width] = np.multiply((y_hat - y), self._grad_g_i[layer](net))
                    temp_grad_E = np.outer(self._delta[layer], self._out_i[layer-1])
                    grad_E[layer] += temp_grad_E
                else:
                    #Non output layer
                    net = self._net_i[layer]
                    partly = np.dot(self._delta[layer+1], np.transpose(self.w[layer+1]))
                    temp_delta = np.multiply(partly, self._grad_g_i[layer](net))
                    self._delta[layer] = temp_delta
                    temp_grad_E = np.outer(self._delta[layer], self._out_i[layer-1])
                    grad_E[layer] += temp_grad_E

        return grad_E

    def train(self,
              x=None,
              y=None,
              reset = False,
              plot = False,
              verbose = False):

        assert x is not None or self.x_train is not None
        assert y is not None or self.y_train is not None
        assert (x is None) == (y is None)
        assert isinstance(reset, bool)
        assert isinstance(plot, bool)
        assert isinstance(verbose, bool)

        gd = GradientDescent(self.j, self.grad_j, x, y, self.w)



        return 0

    def _backprop(self, y, y_hat):
        pass


if __name__ == '__main__':
    # For debugging purpose with the same seed:
    np.random.seed(44)
    plt.ion()

    n_samples   = 500
    d_input     = 1
    d_output    = 1

    def func(x):
        shift = 0.1
        if x > shift:
            return 0.7 * (x-shift) + 0.7
        else:
            return -1.5 * (x-shift) + 0.7

    x = np.linspace(-3., 3., n_samples)[np.newaxis].transpose()
    y = np.zeros(x.shape)

    for i in xrange(0,len(x)):
        y[i] = func(x[i])

    network = NeuralNetwork(1, 5, 1, 1,
                            hidden_neuron_type=reluNeuron,
                            output_neuron_type=linearNeuron)

    network.train(x, y, plot=True, verbose=True)
    network.printWeights()
    print network.w

    plt.show()

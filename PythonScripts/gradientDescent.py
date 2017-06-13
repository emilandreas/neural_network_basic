import copy
import numpy as np
import numpy.random as rand
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from colorama import Fore, Back, Style


class GradientDescent:
    def __init__(self,
                 j, grad_j, eval,        # Cost function
                 x, y,              # Training data
                 theta,             # Cost function parameter
                 b,
                 parameter=None,    # Optimization Hyperparameter
                 verbose=False,
                 network=None):

        # Training and Testing dataset:
        self.p_test  = 0.1
        self.p_val   = 0.1
        self.p_train = 0.8
        self.n_test  = 0
        self.n_val   = 0
        self.n_train = 0

        # Stopping Criteria:
        # - Training stops if j converges below j_min
        # - Training stops if maximum iterations are exceeded
        # - Training stops if performance of training and validation dataset diverge (i.e. early stopping)
        self.j_min = 1e-6
        self.max_epoch = 1000
        self.early_stopping_epochs = 20

        # Gradient approximation:
        # - gradient descent,               i.e. nBatch = nTrain
        # - stochastic gradient descent     i.e. nBatch = 1
        # - mini-batch gradient descent     i.e. nBatch = 2 - n << nTrain
        self.n_batch = 3

        # Adaptive Learning Rates:
        # - fixed alpha i.e. fixed learning rate for iterations and layer
        # - (AdaGrad)   i.e. normalizes learning rates w.r.t. to historic amplitude of gradients
        # - RMSProp     i.e. exponentially smoothed AdaGrad
        # - Adam        i.e. unbiased RMSProp
        self.learning_rate_adaption = self.fixed_alpha
        self.alpha = 1e-3                   # Default Value = 5e-3
        self.adapt_learn_rate_rho = 0.99    # Default Value = 0.999
        self.adapt_learn_rate_delta = 1e-8  # Default Value = 1e-8
        self.adapt_learn_rate_r = []        # Default Initialization = 0.0

        # Momentum:
        # - zero Momentum       i.e. dW_k+1 = alpha * dW_k+1
        # - standard Momentum   i.e. dW_k+1 = rho * dW_k - alpha * dW_k+1
        # - Adam                i.e. dW_k+1 = - alpha * 1/(1-rho) (rho * dW_k + (1-rho) * dW_k+1)
        # - (Nestorov Momentum) i.e. evaluation of gradient @ w = w + v
        self.momentum = self.zero_momentum
        self.momentum_rho = 0.90  # Default Value = 0.9
        self.momentum_v = theta * 0
        self.momentum_v_b = b * 0

        # Verification and unpacking of the input:

        # Cost Functions:
        assert callable(j)
        assert callable(grad_j)

        self.j       = j
        self.grad_j  = grad_j
        self.eval    = eval

        # Training Data and cost function parameter:
        self._dataIn = x
        self._dataLabel = y
        assert x.shape[0] == y.shape[0]
        #shuffle dataset!
        shuffle_vec = np.arange(x.shape[0])
        np.random.shuffle(shuffle_vec)
        self._training_x = x[shuffle_vec]     # Contains the x data used for training
        self._training_y = y[shuffle_vec]     # Contains the y data used for training
        self._theta      = theta # Contains the optimizable parameter of the function
        self._b          =b

        # Optimization Hyperparameters:
        if parameter is not None:

            assert parameter['SGD']['alpha'] <= 1.0
            assert parameter['SGD']['batch size'] <= x.shape[0]
            assert np.isclose(parameter['SGD']['p train'] +
                              parameter['SGD']['p test']  +
                              parameter['SGD']['p val'], 1.0)

            self.alpha      = parameter['SGD']['alpha']
            self.j_min      = parameter['SGD']['j min']
            self.max_epoch  = parameter['SGD']['max epoch']
            self.n_batch    = parameter['SGD']['batch size']
            self.p_train    = parameter['SGD']['p train']
            self.p_test     = parameter['SGD']['p test']
            self.p_val      = parameter['SGD']['p val']

            if parameter['Momentum']['select']:

                assert parameter['Momentum']['select'] is not parameter['Adam']['select']
                assert parameter['Momentum']['rho'] < 1.0

                self.momentum = self.standard_momentum
                self.momentum_rho = parameter['Momentum']['rho']

            if parameter['rmsProp']['select']:

                assert parameter['rmsProp']['select'] is not parameter['Adam']['select']
                assert parameter['rmsProp']['rho'] < 1.0

                self.learning_rate_adaption = self.rmsprop
                self.adapt_learn_rate_rho = parameter['rmsProp']['rho']

            if parameter['Adam']['select']:

                assert parameter['Adam']['beta 1']  < 1.0
                assert parameter['Adam']['beta 2']  < 1.0

                self.learning_rate_adaption = self.adam
                self.momentum = self.adam_momentum
                self.momentum_rho = parameter['Adam']['beta 1']

        self.network = network

        # Initialization of internal variables:
        self._idx_train = []    # Contains the index of samples used for training
        self._idx_test = []     # Contains the index of samples used for testing
        self._idx_val = []      # Contains the index of samples used for validation

        self._j_train = np.zeros(self.early_stopping_epochs)
        self._j_test = np.zeros(self.early_stopping_epochs)
        self._j_val = np.zeros(self.early_stopping_epochs)


        self.slice_dataset(self._training_x.shape[0], self.p_test, self.p_val, self.p_train)
        plt.figure(1)
        f, axarr = plt.subplots(5,5)
        plotRow = 0
        plotCol = 0
        axarr[plotRow, plotCol].plot(self.estimate(), 'r')
        axarr[plotRow, plotCol].plot(self._dataLabel, 'y')
        axarr[plotRow, plotCol].set_title(str(plotRow*4 +plotCol))
        plotCol += 1

        #Training
        num_batches = self.n_train//self.n_batch
        for epoch in xrange(self.early_stopping_epochs):
            for batch in range(num_batches):
                start = batch*self.n_batch
                stop = (batch+1)*self.n_batch
                if batch == num_batches -1:
                    stop = self.n_train
                idx_train_batch = self._idx_train[start:stop]

                delta_w, delta_b= self.grad_j(self._training_x[idx_train_batch], self._training_y[idx_train_batch])
                #self._theta += self.fixed_alpha(delta_w)
                #self._b     += self.fixed_alpha(delta_b)
                delta_w, delta_b = self.standard_momentum(self.alpha, delta_w, delta_b)
                self._theta += delta_w
                self._b += delta_b

            self._j_train[epoch] = self.get_mean_err(self._training_x[self._idx_train], self._training_y[self._idx_train])
            self._j_test[epoch] = self.get_mean_err(self._training_x[self._idx_test], self._training_y[self._idx_test])
            self._j_val[epoch] = self.get_mean_err(self._training_x[self._idx_val], self._training_y[self._idx_val])
            axarr[plotRow, plotCol].plot(self.estimate(), 'r')
            axarr[plotRow, plotCol].plot(self._dataLabel, 'y')
            axarr[plotRow, plotCol].set_title(str(plotRow*4 +plotCol))
            if plotCol == 4:
                plotCol = 0
                plotRow += 1
            else:
                plotCol += 1

        #plt.show()
        #
        #
        #
        #
        # Fill in
        #
        #
        #
        #
        #
        plt.figure(2)
        plt.subplot(211)
        self.plot_err()
        plt.subplot(212)
        self.print_estimation()
        plt.show()


        # print "\n\n"
        # print "########################################################################################################"
        # print "Optimization terminated\n"
        # print "Stopping Criteria \t= {0}".format(self.results['stop_criteria'])
        #
        # print "Cost function \t\t= {0:.1e} / {1:.1e} / {2:.1e}".format(self.results['j_train'],
        #                                                                self.results['j_val'],
        #                                                                self.results['j_test'])
        #
        # print "Iteration \t\t\t= {0:04d}/{1:04d}".format(self.results['epochs'],
        #                                                  self.results['max epochs'])
        # print "########################################################################################################"
        # print "\n\n"

        return

    def slice_dataset(self, n_total, p_test, p_val, p_train):
        self.n_test  = int(n_total*p_test)  # Fill In
        self.n_val   = int(n_total*p_val)  # Fill In
        self.n_train = n_total - self.n_test - self.n_val  # Fill In

        assert (self.n_test + self.n_val + self.n_train) == n_total
        assert (self.n_test > 0 and self.n_val > 0 and self.n_train > 0)


        self._idx_val = np.array(range(0, self.n_val))
        self._idx_test = np.array(range(self.n_val, self.n_val+self.n_test))
        self._idx_train = np.array(range(self.n_val+self.n_test, n_total))

        assert np.intersect1d(self._idx_val, np.intersect1d(self._idx_test, self._idx_train)).size == 0
        assert np.all(
            np.sort(np.union1d(self._idx_val, np.union1d(self._idx_test, self._idx_train))) == np.arange(n_total))


    def fixed_alpha(self, grad_j):
        return -self.alpha * grad_j

    def rmsprop(self, grad_j):
        return np.nan

    def adam(self, grad_j):
        return np.nan

    def standard_momentum(self, alpha, grad_j, grad_j_b):
        self.momentum_v = self.momentum_rho*self.momentum_v - alpha*grad_j
        self.momentum_v_b = self.momentum_rho*self.momentum_v_b - alpha*grad_j_b
        return self.momentum_v, self.momentum_v_b

    def adam_momentum(self, alpha, grad_j):
        return np.nan

    def zero_momentum(self, alpha, grad_j):
        return np.nan

    def plot_err(self):
        plt.plot(self._j_test, 'b')
        plt.plot(self._j_train, 'r')
        plt.plot(self._j_val, 'g')
        train_patch = mpatches.Patch(color='red', label='Training data')
        test_patch = mpatches.Patch(color='blue', label='Testing data')
        val_patch = mpatches.Patch(color='green', label='Validation data')
        plt.legend(handels=[train_patch, test_patch, val_patch])

        return np.nan
    def get_mean_err(self, x, y):
        (rows, columns) = x.shape
        assert columns == 1
        err = 0
        for itr in range(rows):
            err += self.j(x[itr], y[itr])
        return float(err)/rows
    def estimate(self):
        num = self._dataIn.shape[0]
        y_hat = np.zeros(num)
        for i in range(num):
            y_hat[i] = self.eval(self._dataIn[i])[:1]
        return y_hat


    def print_estimation(self):
        y_hat = self.estimate()
        plt.plot(y_hat, 'y')
        plt.plot(self._dataLabel, 'r')


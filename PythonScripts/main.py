from NeuralNetwork import *
#from HyperParameterOptimization import *


def piecewise_linear(x_range, n_points):

    m_1 = -0.7
    m_2 = 1.5
    v_trans = 0.7
    h_trans = 0.5

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.zeros(x.shape)

    y[x >= h_trans] = m_1 * (x[x >= h_trans]- h_trans) + v_trans
    y[x < h_trans]  = m_2 * (x[x < h_trans] - h_trans)  + v_trans

    return x.reshape(n_points, 1), y.reshape(n_points, 1)


def polynom(x_range, n_points):
    a = +0.1
    b = +0.0
    c = -0.695
    v_trans = 0.2
    h_trans = 0.0

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = a * (x-h_trans)**3 + b * (x-h_trans)**2 + c * (x-h_trans) + v_trans

    return x.reshape(n_points, 1), y.reshape(n_points, 1)

def trigonometric(x_range, n_points):
    w = 6 * np.pi
    amp = +1.0
    m = +1.0
    v_trans = 0.0
    h_trans = 0.4

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = amp * (x-h_trans) * np.cos(w * (x-h_trans)) + m * (x-h_trans) + v_trans

    return x.reshape(n_points, 1), y.reshape(n_points, 1)

def regression():
    # Generate Data
    x_range = (-3., +3.)
    n_points = 500

    # Change the function to piecewise_linear / polynom / trigonometric
    x, y = polynom(x_range, n_points)

    # Create Neural Network
    n_hidden_layer = 10
    n_hidden_units = 20

    network = NeuralNetwork(1, n_hidden_units, n_hidden_layer, 1,
                            hidden_neuron_type=reluNeuron,
                            output_neuron_type=linearNeuron)

    network.set_training_data(x, y)
    network.train(plot=True, verbose=True)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    regression()

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivada(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivada(x):
    return 1.0 - x ** 2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # inicializo los pesos
        self.weights = []
        self.deltas = []
        # capas = [5, 7, 3, 1]
        # random de pesos varia entre (-1, 1)
        # asigno valores aleatorios a capa de entrada y capa/s oculta/s
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # asigno aleatorios a capa de salida
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, x, y, learning_rate=0.2, epochs=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias a la capa de entrada
        ones = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((ones.T, x), axis=1)

        for k in range(epochs):
            i = np.random.randint(x.shape[0])
            a = [x[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # Calculo la diferencia en la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # Empezamos en el segundo layer hasta el ultimo
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
            self.deltas.append(deltas)

            # invertir
            # [levelN(output)->levelN-1(hidden)]  => [levelN-1(hidden)->levelN(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiplicar los delta de salida con las activaciones de entrada
            #    para obtener el gradiente del peso.
            # 2. Actualizar el peso restandole un porcentaje del gradiente
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 100000 == 0:
                print('epochs:', k)

    def predict(self, x):
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        print("LISTADO PESOS DE CONEXIONES")
        for i in range(len(self.weights)):
            print(self.weights[i])

    def savetxt(self, filename):
        for i in range(len(self.weights)):
            np.savetxt('./salidas_pesos/layer_'+str(i+1)+'_'+filename, self.weights[i], fmt='%.60f')

    def loadtxt(self, filename):
        pesos = []
        for i in range(len(self.weights)):
            pesos.append(np.loadtxt('./salidas_pesos/layer_'+str(i+1)+'_'+filename, ndmin=2))
        self.weights = pesos

    def get_deltas(self):
        return self.deltas

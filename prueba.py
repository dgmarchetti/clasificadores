from NeuralNetwork import NeuralNetwork
import data_io as io


def binarizar(p):
    if p >= 0.5:
        return 1
    else:
        return 0


layers = [5, 7, 3, 1]

# Datos TP
nn = NeuralNetwork(layers)
x = io.get_x_data('test').values
y = []
# Cargar los pesos obtenidos en el entrenamiento
nn.loadtxt('test.out')

# Predecir para cada vector X de entrada
for e in x:
    predicted = nn.predict(e)
    y.append(binarizar(predicted[0]))

# Guardar predicciones en archivo de salida
io.save_y_data(y)

# # Prueba de accuracy con los valores de entrenamiento con la red ya entrenada
# def correcto(s, p):
#     if p > 0.5 and s == 1:
#         return 1
#     elif p <= 0.5 and s == 0:
#         return 1
#     return 0
#
#
# index = 0
# correctos = 0
# # Datos TP
# nn = NeuralNetwork(layers)
# x = io.get_x_data().values
# y = io.get_y_data().values
# # Cargar los pesos obtenidos en el entrenamiento
# nn.loadtxt('test.out')
#
# for e in x:
#     predicted = nn.predict(e)
#     bien = correcto(y[index][0], predicted[0])
#     correctos += bien
#     # print("Entradas:", e, "Salida:", y[index], "Predicción:", predicted, "Bien: ", bien, "Correctos:", correctos)
#     index += 1
#
# print("Correctos " + str(correctos) + " sobre " + str(index) + " "+ str(correctos/index*100) + "%.")



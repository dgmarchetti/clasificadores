# import numpy as np
# import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import data_io as io

def correcto(s, p):
    if p > 0.5 and s == 1:
        return 1
    elif p <= 0.5 and s == 0:
        return 1
    return 0

layers = [5, 7, 3, 1]
nro_epochs = 4000000
index = 0
correctos = 0
tasa_aprendizaje = 0.02

# Datos TP
nn = NeuralNetwork(layers)
x = io.get_x_data().values
y = io.get_y_data().values
nn.fit(x, y, learning_rate=tasa_aprendizaje, epochs=nro_epochs)

for e in x:
    predicted = nn.predict(e)
    bien = correcto(y[index][0], predicted[0])
    correctos += bien
    print("Entradas:", e, "Salida:", y[index], "Predicción:", predicted, "Bien: ", bien, "Correctos:", correctos)
    index += 1

print("Layers:", layers, "Epochs:",nro_epochs, "Learning Rate:", tasa_aprendizaje)
print("Correctos " + str(correctos) + " sobre " + str(index) + " "+ str(correctos/(index)*100) + "%.")


# Imprimir los pesos de las conexiones
nn.print_weights()

# Graficamos la función coste
# deltas = nn.get_deltas()
# valores=[]
# index=0
# for arreglo in deltas:
#     valores.append(arreglo[1][0] + arreglo[1][0])
#     index=index+1
#
# plt.plot(range(len(valores)), valores, color='b')
# plt.ylim([0, 1])
# plt.ylabel('Cost')
# plt.xlabel('Epochs')
# plt.tight_layout()
# plt.show()
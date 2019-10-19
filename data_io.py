import pandas as pd
data_file_Xtrain = "./datasets/train/X_train.csv"
data_file_Ytrain = "./datasets/train/Y_train.csv"

# Create a list of the new column labels: new_labels
Xlabels = ['X1', 'X2', 'X3', 'X4', 'X5']
Ylabels = ['Y']

# Read in the file, specifying the header and names parameters: xTrain
xTrain = pd.read_csv(data_file_Xtrain, names=Xlabels)
yTrain = pd.read_csv(data_file_Ytrain, names=Ylabels)

# Print both the DataFrames head and info
# print(xTrain.head(10))
print(xTrain.info())
# print(yTrain.head(10))
print(yTrain.info())
# print(xTrain.describe())

# Armar vector patrón aumentado
# xTrain['W6'] = 1
# print(xTrain.head(10))


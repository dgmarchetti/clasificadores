import pandas as pd


def get_x_data():
    data_file_xtrain = "./datasets/train/X_train.csv"
    # Create a list of the new column labels: new_labels
    xlabels = ['X1', 'X2', 'X3', 'X4', 'X5']
    # Read in the file, specifying the header and names parameters: xtrain
    xtrain = pd.read_csv(data_file_xtrain, names=xlabels)
    # print(xtrain.head(10))
    # print(xtrain.info())
    return xtrain


def get_y_data():
    data_file_ytrain = "./datasets/train/Y_train.csv"
    # Create a list of the new column labels: new_labels
    ylabels = ['Y']
    # Read in the file, specifying the header and names parameters: ytrain
    ytrain = pd.read_csv(data_file_ytrain, names=ylabels)
    # print(ytrain.head(10))
    # print(ytrain.info())
    return ytrain

import pandas as pd


def get_x_data(data='train'):
    if data=='train':
        data_file_xtrain = "./datasets/train/X_train.csv"
    elif data=='test':
        data_file_xtrain = "./datasets/test/X_test.csv"
    # Create a list of the new column labels: new_labels
    xlabels = ['X1', 'X2', 'X3', 'X4', 'X5']
    # Read in the file, specifying the header and names parameters: xtrain
    xtrain = pd.read_csv(data_file_xtrain, names=xlabels)
    return xtrain


def get_y_data():
    data_file_ytrain = "./datasets/train/Y_train.csv"
    # Create a list of the new column labels: new_labels
    ylabels = ['Y']
    # Read in the file, specifying the header and names parameters: ytrain
    ytrain = pd.read_csv(data_file_ytrain, names=ylabels)
    return ytrain

def save_y_data(data):
    df = pd.DataFrame(data)
    df.to_csv("./datasets/test/Y_test.csv",index=False,header=False)

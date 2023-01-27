# Cashton Holbert
import numpy as np
import math

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):

    train_data = np.loadtxt(Xtrain_file, delimiter=",", dtype=float)
    train_labels = np.loadtxt(Ytrain_file, dtype=float)
    test_data = np.loadtxt(test_data_file, delimiter=",", dtype=float)

    # change the labels to work with the np.sign() function
    for i in range(len(train_labels)):
        if int(train_labels[i]) == 0:
            train_labels[i] = -1

    # initialize the variables 
    epochs = 1
    k = 0 #index representation for vector v and weights c
    v = np.zeros((len(train_data), len(train_data[0]))) # vector
    c = np.zeros(len(train_labels)) # list of weights corresponding to v

    # train the model
    for t in range(epochs + 1):
        for i in range(len(train_data)):
            
            # make "prediction", if wrong, update k and update v[k+1] and c[k+1], else update current c[k]
            if train_labels[i] * np.dot(v[k], train_data[i]) <= 0:  
                v[k+1] = v[k] + (train_labels[i] * train_data[i])
                c[k+1] = 1
                k = k + 1
            else:
                c[k] += 1

    # now make predictions on the model given test_data_file
    pred_array = []
    for i in range(len(test_data)):
        temp_sum = 0
        for j in range(k):
            temp_sum += c[k] * np.sign(np.dot(v[k], test_data[i]))
        pred_array.append(np.sign(temp_sum))

    # change labels to match specified output, so that means if labeled -1 because of np.sign() function, label it as 0 instead
    for i in range(len(pred_array)):
        if int(pred_array[i]) == -1:
            pred_array[i] = 0
    
    # save predictions into new pred_file.csv
    np.savetxt(pred_file, pred_array, fmt='%1d', delimiter=",")        


if __name__ == "__main__":
    xtrain = 'Xtrain.csv'
    ytrain = 'Ytrain.csv'
    test = 'test_file.csv'
    pred = 'pred_file.csv'
    run(xtrain, ytrain, test, pred)
    
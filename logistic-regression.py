import csv
import numpy as np
import matplotlib.pyplot as plt

def open_csv(csv_filename, prepend_ones):
    X1list = []
    X2list = []
    ylist = []
    with open(csv_filename, 'rt') as csvfile:
        readfile = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(readfile):
            X1list.append(row[0])
            X2list.append(row[1])
            ylist.append(row[2])
    X1list = np.array(map(float,X1list))
    X2list = np.array(map(float,X2list))
    ylist = np.array(map(float,ylist))
    X_all = np.column_stack((X1list, X2list))
    if prepend_ones == True:
        ones = np.ones(shape=ylist.shape)[..., None]
        X_all = np.concatenate((ones, X_all), 1)
    return (X_all, ylist) 

if __name__ == "__main__":
    print("Logistic Regression")
    (cl_train_x, cl_train_results) = open_csv('cl_train_1.csv', False)
    (cl_test_x, cl_test_results) = open_csv('cl_test_1_csv', False)
    print(cl_test_x)
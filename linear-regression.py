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
            if idx == 0:
                continue
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

# def plot_graph(xvalues, yvalues, predictions):
#     plt.scatter(xvalues, yvalues)
#     plt.title('Graph')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()

def mean_squared_error(predictions, test_data_results, weights):
    n = len(predictions)
    total_error = 0
    for x in range(0,n):
        total_error += (predictions[x] - test_data_results[x])**2
    mse = 1.0/n * total_error
    print "Mean Squared Error:", mse
    return mse

def linear_regression(weights, test_data, test_data_results):
    predictions = []
    for i in test_data:
        components = weights[1:] * i
        predictions.append(sum(components) + weights[0])
    mean_squared_error(predictions, test_data_results, weights)
    return predictions


def closed_form(x_data, y_data):
    w = np.linalg.pinv(x_data.transpose().dot(x_data)).dot(x_data.transpose()).dot(y_data)
    print "Weights:", w
    return w

if __name__ == "__main__":
    (training_set_x, training_results_y) = open_csv('train_2d_reg_data.csv', True)
    (test_set_x, test_results_y) = open_csv('test_2d_reg_data.csv', False)
    linear_regression(closed_form(training_set_x, training_results_y),test_set_x, test_results_y)
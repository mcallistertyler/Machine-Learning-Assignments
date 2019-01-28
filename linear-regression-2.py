import csv
import numpy as np
import matplotlib.pyplot as plt

def open_csv(csv_filename, prepend_ones):
    X1list = []
    ylist = []
    with open(csv_filename, 'rt') as csvfile:
        readfile = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(readfile):
            if idx == 0:
                continue
            X1list.append(row[0])
            ylist.append(row[1])
    X1list = np.array(map(float,X1list))
    ylist = np.array(map(float,ylist))
    if prepend_ones == True:
        ones = np.ones(shape=ylist.shape)[..., None]
        X1list = np.column_stack((ones, X1list))
    return (X1list, ylist)            

def plot_graph(xvalues, yvalues, predictions):
    print xvalues
    print predictions
    fit = np.polyfit(xvalues,yvalues,1)
    fit_fn = np.poly1d(fit)
    plt.plot(xvalues,yvalues,'yo',xvalues,fit_fn(predictions),'--k')
    #plt.scatter(xvalues, yvalues)
    plt.title('ML&CBR Assignment 1')
    plt.xlabel('test_set_x')
    plt.ylabel('test_results_y')
    plt.show()

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
    plot_graph(test_data, test_data_results, predictions)
    return predictions


def closed_form(x_data, y_data):
    w = np.linalg.pinv(x_data.transpose().dot(x_data)).dot(x_data.transpose()).dot(y_data)
    print "Weights:", w
    return w

if __name__ == "__main__":
    (training_set_x, training_results_y) = open_csv('train_1d_reg_data.csv', True)
    (test_set_x, test_results_y) = open_csv('test_1d_reg_data.csv', False)
    linear_regression(closed_form(training_set_x, training_results_y),test_set_x, test_results_y)
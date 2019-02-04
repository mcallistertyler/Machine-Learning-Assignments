import csv
import numpy as np
import matplotlib.pyplot as plt
def open_csv(csv_filename, prepend_ones):
    X1list = []
    X2list = []
    X3list = []
    X4list = []
    ylist = []
    with open(csv_filename, 'rt') as csvfile:
        readfile = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(readfile):
            X1list.append(row[0])
            X2list.append(row[1])
            ylist.append(row[2])
    X1list = np.array(map(float,X1list))
    X2list = np.array(map(float,X2list))
    X3list = np.square(X1list)
    X4list = np.square(X2list)
    ylist = np.array(map(float,ylist))
    X_all = np.column_stack((X1list, X2list, X3list, X4list))
    ## prepend ones to the data. this is the bias
    if prepend_ones == True:
        ones = np.ones(shape=ylist.shape)[..., None]
        X_all = np.concatenate((ones, X_all), 1)
    return (X_all, ylist) 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def classify(prediction):
    classifications = []
    for x in range(0,len(prediction)):
        if prediction[x] >= 0.5:
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications

def predict(x_data, weights):
    z = np.dot(x_data, weights)
    return sigmoid(z)

def gradient_descent(x_data, targets, weights, alpha):
    predictions = predict(x_data, weights)
    gradient = np.dot(x_data.transpose(), (predictions - targets)) / targets.shape[0]
    weights -= alpha * gradient    
    return weights

def cross_entropy(x_data, y_data, weights):
    observations = len(y_data)
    predictions = predict(x_data, weights)

    class1_cost = -y_data*np.log(predictions)
    class2_cost = (1-y_data)*np.log(1-predictions)

    cost = class1_cost - class2_cost

    cost = cost.sum()/observations

    return cost

def train(x_data, y_data, weights, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        weights = gradient_descent(x_data, y_data, weights, alpha)
        cost = cross_entropy(x_data, y_data, weights)
        cost_history.append(cost)
    return weights, cost_history

def plot_cost(costs):
    fig = plt.figure()
    plt.plot(costs)

def plot_graph(xvalues, yvalues, predictions, weights, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fit_fn = -(weights[1] * xvalues[:,1])/weights[2]-weights[0]/weights[2]
    plt.plot(xvalues[:,1],fit_fn, 'k-')
    for x in range(0, xvalues.shape[0]):
        if(predictions[x] < 0.5):
            ax.scatter(xvalues[:,1][x], predictions[x], c='b', marker="o")
        else:
            ax.scatter(xvalues[:,1][x], predictions[x], c='r', marker="x")
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')

if __name__ == "__main__":
    (cl_train_x, cl_train_results) = open_csv('cl_train_2.csv', True)
    (cl_test_x, cl_test_results) = open_csv('cl_test_2.csv', True)
    init_weights = np.random.uniform(-1, 1, (5,))
    (trained_weights, cost_history) = train(cl_train_x, cl_train_results, init_weights, 0.6, 6000)
    print "----------Training----------"
    print "Trained predictions", predict(cl_train_x, trained_weights)
    print "Trained classifications", classify(predict(cl_test_x, trained_weights))
    print "Trained cost", cost_history[-1]
    plot_cost(cost_history)
    print "Trained weights", trained_weights
    #plot_graph(cl_train_x, cl_train_results, predict(cl_train_x, trained_weights), trained_weights, "Programming Task 2 - Training Set 2")
    print "-----------------------------\n"
    print "----------Testing------------"
    print "Test predictions", predict(cl_test_x, trained_weights)
    print "Test classifications", classify(predict(cl_test_x, trained_weights))
    print "Test cost", cross_entropy(cl_test_x, cl_test_results, trained_weights)
    print "-----------------------------\n"
    #plot_graph(cl_test_x, cl_test_results, predict(cl_test_x, trained_weights), trained_weights, "Programming Task 2 - Test Set 1")
    plt.show()
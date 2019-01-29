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

def sigmoid(z):
    return 1/(1+np.exp(-z))

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

def classification(probability):
    probability = np.vectorize(probability)
    # decision_boundary = np.vectorize(decision_boundary)
    # print decision_boundary(probability).flatten()
    total_classifications = []
    if probability >= 0.5:
        total_classifications.append(1)
    else:
        total_classifications.append(0)
    #print(total_classifications)

def predict(features, weights):
    z = np.dot(features, weights)
    #print "Prediction :", sigmoid(z)
    classification(sigmoid(z))
    print "Classification:", sigmoid(z)
    return sigmoid(z)

def gradient_descent(features, targets, weights, alpha):
    N = len(features)
    predictions = predict(features, weights)
    gradient = np.dot(features.transpose(), predictions - targets)
    gradient /= N
    gradient *= alpha
    weights -= gradient

    return weights

def cross_entropy(features, labels, weights):
    observations = len(labels)
    predictions = predict(features, weights)

    class1_cost = -labels*np.log(predictions)
    class2_cost = (1-labels)*np.log(1-predictions)

    cost = class1_cost - class2_cost

    cost = cost.sum()/observations

    return cost

def train(features, labels, weights, alpha, iteratons, test_mode):
    cost_history = []
    if test_mode == False:
        for i in range(iteratons):
            weights = gradient_descent(features, labels, weights, alpha)

            cost = cross_entropy(features, labels, weights)
            cost_history.append(cost)

            if i % 100 == 0:
                print "Iteration: " + str(i) + " Cost: " + str(cost)
                print "Weights: ", weights
                print "Cost history: ", cost_history[i]
    else:
        cost = cross_entropy(features, labels, weights)
        cost_history.append(cost)
    return weights, cost_history

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff))/len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right')
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()

if __name__ == "__main__":
    (cl_train_x, cl_train_results) = open_csv('cl_train_1.csv', False)
    (cl_test_x, cl_test_results) = open_csv('cl_test_1.csv', False)
    init_weights = np.random.uniform(-2, 2, (2,))
    (trained_weights, cost_history) = train(cl_train_x, cl_train_results, init_weights, 0.02, 6000, False)
import sys
import numpy as np


def compare(predictions):
    resultsVector = np.loadtxt(sys.argv[2])
    correct = 0
    fail = 0
    for i in range(len(predictions)):
        if predictions[i] == resultsVector[i]:
            correct += 1
        else:
            fail += 1
    successRate = (correct / (fail + correct)) * 100
    print(successRate, "%")

def zScoreNorma(data_set):
    return (data_set - data_set.mean(0)) / data_set.std(0)

def dist(a, b):
    d = np.linalg.norm(a - b)
    return d

# output of functions is array of predictions
def knn():
    k = 30
    trainPoints = np.loadtxt(sys.argv[1], delimiter=",")
    resultsVector = np.loadtxt(sys.argv[2])
    testPoints = np.loadtxt(sys.argv[3], delimiter=",")
    # compute distance for each point
    distancesForTestPoints = []
    # iterate over test rows (test points)
    for i in range(len(testPoints)):
        distVector = []
        distVector.clear()
        for j in range(len(trainPoints)):
            d = dist(testPoints[i], trainPoints[j])
            distVector.append((j, d))
        distancesForTestPoints.append(distVector)
    # sort distances
    predictionsVector = []
    for i in range(len(distancesForTestPoints)):
        classifications = []
        classifications.clear()
        distancesForTestPoints[i].sort(key=lambda tup: tup[1])
        for j in range(k):
            nearestPoint = distancesForTestPoints[i][j][0]
            num = resultsVector[nearestPoint]
            classifications.append(num)
        # find the most common classification
        prediction = max(set(classifications), key=classifications.count)
        predictionsVector.append(prediction)
    return predictionsVector

def perceptron():
    lr = 0.3
    numOfEpocs = 30
    X = zScoreNorma(np.loadtxt(sys.argv[1], delimiter=","))
    Y = resultsVector = np.loadtxt(sys.argv[2])
    test = zScoreNorma(np.loadtxt(sys.argv[3], delimiter=","))
    n_samples, n_features = X.shape
    w = np.zeros((3, X.shape[1]))
    for x in range(numOfEpocs):
        for x_i, y_i in zip(X, Y):
            y_hat = np.argmax(np.dot(w, x_i))
            y_i = int(y_i)
            y_hat = int(y_hat)
            if y_i != y_hat:
                update = lr * x_i
                w[y_i, :] = w[y_i, :] + update
                w[y_hat, :] = w[y_hat, :] - update
        if x % 10 == 1:
            lr *= 0.1
    predictions = []
    for sample in test:
        y_hat = np.argmax(np.dot(w, sample))
        predictions.append(int(y_hat))
    return predictions

def svm():
    lr = 0.3
    numOfEpocs = 30
    delta = 0.01
    X = zScoreNorma(np.loadtxt(sys.argv[1], delimiter=","))
    Y = resultsVector = np.loadtxt(sys.argv[2])
    test = zScoreNorma(np.loadtxt(sys.argv[3], delimiter=","))
    n_samples, n_features = X.shape
    w = np.zeros((3, X.shape[1]))
    for x in range(numOfEpocs):
        for x_i, y_i in zip(X, Y):
            y_hat = np.argmax(np.dot(w, x_i))
            y_i = int(y_i)
            y_hat = int(y_hat)
            loss = 1 - np.dot(x_i, w[y_i]) + np.dot(x_i, w[y_hat])
            if 0 < loss:
                deltaLr = 1 - (delta * lr)
                lrUpdate = lr * x_i
                w[y_i] = deltaLr * w[y_i] + lrUpdate
                w[y_hat] = deltaLr * w[y_hat] - lrUpdate
                for i in range(len(w)):
                    if i != y_i and i != y_hat:
                        w[i] = w[i] * deltaLr
            else:
                w = w * (1 - delta * lr)
        if x % 10 == 1:
            lr *= 0.1
    predictions = []
    for sample in test:
        y_hat = np.argmax(np.dot(w, sample))
        predictions.append(int(y_hat))
    return predictions

def pa():
    numOfEpocs = 40
    X = zScoreNorma(np.loadtxt(sys.argv[1], delimiter=","))
    Y = resultsVector = np.loadtxt(sys.argv[2])
    test = zScoreNorma(np.loadtxt(sys.argv[3], delimiter=","))
    n_samples, n_features = X.shape
    w = np.zeros((3, X.shape[1]))
    for x in range(numOfEpocs):
        for x_i, y_i in zip(X, Y):
            y_hat = np.argmax(np.dot(w, x_i))
            y_i = int(y_i)
            y_hat = int(y_hat)
            loss = 1 - np.dot(x_i, w[y_i]) + np.dot(x_i, w[y_hat])
            dist = np.linalg.norm(x_i) * np.linalg.norm(x_i)
            tau = loss if loss > 0 else 0
            tau /= 2 * dist
            if y_i != y_hat:
                w[y_i] = w[y_i] + tau * x_i
                w[y_hat] = w[y_hat] - tau * x_i
    predictions = []
    for sample in test:
        y_hat = np.argmax(np.dot(w, sample))
        predictions.append(int(y_hat))
    return predictions

if __name__ == "__main__":
    # compare(knn())
    # compare(perceptron())
    # compare(svm())
    # compare(pa())

    knnVector = knn()
    perceptronVector = perceptron()
    svmVector = svm()
    paVector = pa()
    output_file = open(sys.argv[4], "w")
    for knn, perceptron, svm, pa in zip(knnVector, perceptronVector, svmVector, paVector):
        line = "knn: " + str(int(knn)) + ", perceptron: " + str(int(perceptron)) + ", svm: " + str(int(svm)) + ", pa: " + str(int(pa)) + "\n"
        output_file.write(line)
    output_file.close()
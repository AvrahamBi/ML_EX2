import sys
import numpy as np

def zscore_normalization(data_set):
    return (data_set - data_set.mean(0)) / data_set.std(0)

def minmax_normalization(data_set):
    min_vector = np.min(data_set, axis=0)
    max_vector = np.max(data_set, axis=0)
    for row in data_set:
        for i in range(len(row)):
            row[i] = (row[i] - min_vector[i]) / (max_vector[i] - min_vector[i])
    return data_set

def dist(a, b):
    d = np.linalg.norm(a - b)
    return d

# output of functions is array of predictions
def knn(k):
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


class Perceptron:
    def __init__(self, X, Y, lr):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.n_samples, self.n_features = X.shape
        self.w = np.random.random((3, X.shape[1])) # todo zeros or random.random

    def train(self, numOfEpocs):
        for x in range(numOfEpocs):
            for x_i, y_i in zip(self.X, self.Y):
                y_hat = np.argmax(np.dot(self.w, x_i))
                y_i = int(y_i)
                y_hat = int(y_hat)
                if y_i != y_hat:
                    update = self.lr * x_i
                    self.w[y_i, :] = self.w[y_i, :] + update
                    self.w[y_hat, :] = self.w[y_hat, :] - update
            if x % 10 == 1:
                self.lr *= 0.1


    def run(self, test):
        predictions = []
        for sample in test:
            y_hat = np.argmax(np.dot(self.w, sample))
            predictions.append(int(y_hat))
        return predictions


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


def Svm():
    pass

def Pa():
    pass

if __name__ == "__main__":
    trainPoints = np.loadtxt(sys.argv[1], delimiter=",")
    resultsVector = np.loadtxt(sys.argv[2])
    testPoints = np.loadtxt(sys.argv[3], delimiter=",")

    trainPoints = zscore_normalization(trainPoints)
    testPoints = zscore_normalization(testPoints)

    p = Perceptron(trainPoints, resultsVector, 0.1)
    p.train(100)
    perceptronPrediction = p.run(trainPoints)
    compare(perceptronPrediction)

    #knnPrediction = knn(3)
    #perceptronPrediction = perceptron()
    #svmPrediction = svm()
    #paPrediction = pa()
    # output_file = open(sys.argv[4], "w")

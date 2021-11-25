import sys
import numpy as np
from scipy import stats

def zscore_normalization(data_set):
    #return (data_set - data_set.mean(0)) / data_set.std(0)
    return stats.zscore(data_set) # todo remove this Scipy

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
    #
    # shufller = np.random.permutation(len(trainPoints))
    # trainPoints = trainPoints[shufller]
    # tresultsVector = resultsVector[shufller]
    #
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

    # Conclusion
    correct = 0
    fail = 0
    for i in range(len(predictionsVector)):
        if predictionsVector[i] == resultsVector[i]:
            correct += 1
        else:
            fail += 1
    successRate = (correct / (fail + correct)) * 100
    print("KNN(" + str(k) +") Rate:", round(successRate, 4))
    return (k, successRate)


def perceptron():
    pass

def svm():
    pass

def pa():
    pass




if __name__ == "__main__":
    x = knn(2)
    for i in range(3, 20, 1):
        y = knn(i)
        if (y[1] > x[1]):
            x = y
    print("")
    print("BEST: " + "KNN(" + str(x[0]) + ") Rate:", str(round(x[1], 4)))
    perceptron()
    svm()
    pa()
    # output_file = open(sys.argv[4], "w")

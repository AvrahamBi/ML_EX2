import sys
import numpy as np


def convertListToFloat(txtPoints):
    points = []
    for txtPoint in txtPoints:
        point = []
        point.clear()
        for val in txtPoint:
            point.append(float(val))
        points.append(point)
    return points

def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    d = np.linalg.norm(a - b)
    return d

# output of functions is array of predictions
def knn(k):
    # python ex2.py <train_x_path> <train_y_path> <test_x_path> <output_log_name>
    train_x_path, train_y_path, test_x_path, output_log_name = sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4]
    # open file
    with open(train_x_path) as file:
        trainPoints = file.readlines()
        trainPoints = [line.rstrip().split(",") for line in trainPoints]
        trainPoints = convertListToFloat(trainPoints)
    file.close()
    # open results file
    with open(train_y_path) as file:
        resultsVector = file.readlines()
        resultsVector = [line.rstrip() for line in resultsVector]
        for i in range(len(resultsVector)):
            resultsVector[i] = int(resultsVector[i])
    file.close()
    # open test file
    with open(test_x_path) as file:
        testPoints = file.readlines()
        testPoints = [line.rstrip().split(",") for line in testPoints]
        testPoints = convertListToFloat(testPoints)
    file.close()

    # compute distance for each point
    distancesForTestPoints = []
    for testPoint in testPoints:
        check = 0
        distVector = []
        distVector.clear()
        for x in range(len(trainPoints)):
            distVector.append((x, dist(trainPoints[x], testPoint)))
        distancesForTestPoints.append(distVector)
    #
    # sort distances
    predictionsVector = []
    for testPoint in distancesForTestPoints:
        testPoint.sort(key=lambda tup: tup[1])
        classifications = []
        classifications.clear()
        for i in range(k):
            p = testPoint[i]
            classifications.append(resultsVector[p[0]])
        # find the most common classification
        prediction = max(set(classifications), key=classifications.count)
        predictionsVector.append(prediction)

    ######## Conclusion
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
    x = knn(4)
    for i in range(1, 21, 1):
        y = knn(i)
        if (y[1] > x[1]):
            x = y
    print("")
    print("BEST: " + "KNN(" + str(x[0]) + ") Rate:", str(round(x[1], 4)))
    perceptron()
    svm()
    pa()

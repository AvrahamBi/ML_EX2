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
    return np.linalg.norm(a - b)

# output of functions is array of predictions
def knn():
    # python ex2.py <train_x_path> <train_y_path> <test_x_path> <output_log_name>
    train_x_path, train_y_path, test_x_path, output_log_name = sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4]
    # open file
    with open(train_x_path) as file:
        txtPoints = file.readlines()
        txtPoints = [line.rstrip().split(",") for line in txtPoints]
    file.close()
    # convert to float
    points = convertListToFloat(txtPoints)


    print("Check")






def perceptron():
    pass

def svm():
    pass

def pa():
    pass
    
if __name__ == "__main__":
    knn()
    perceptron()
    svm()
    pa()

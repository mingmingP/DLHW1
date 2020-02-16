import math
import numpy as np
from download_mnist import load
import operator
import time

# classify using kNN
# x_train = np.load('../x_train.npy')
# y_train = np.load('../y_train.npy')
# x_test = np.load('../x_test.npy')
# y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)


def kNNClassify(newInput, dataSet, labels, k):
    result = []
    ########################
    # Input your code here #
    ########################
    for i in range(0, len(newInput)):
        distance = []
        for j in range(0,len(dataSet)):
            #print("newinput")
            #print(dataSet[i,:,:])
            #print("traindata")
            #print(dataSet[j,:,:])
            distances = np.linalg.norm(dataSet[j,:,:] - newInput[i, :,:])
            distance.append(distances)
        ##print(np.shape(distance))
        kmin_index = np.argpartition(distance, k)
        ##print(i,kmin_index)
        voters = labels[kmin_index[:k]]
        result.append(voters[np.argmax(np.bincount(voters))])
    ##print(result)


    ####################
    # End of your code #
    ####################
    return result


start_time = time.time()
outputlabels = kNNClassify(x_test[0:20], x_train, y_train, 10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result) / len(outputlabels))
print("---classification accuracy for knn on mnist: %s ---" % result)
print("---execution time: %s seconds ---" % (time.time() - start_time))
print(outputlabels)

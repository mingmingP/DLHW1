from collections import Counter

import numpy as np

train_data = [[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 0], [1, 2, 2], [2, 2, 2], [1, 2, -1], [2, 2, 3], [-1, -1, -1],
              [0, -1, -2], [0, -1, 1], [-1, -2, 1]]
train_label = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
train_set = np.array(train_data)
test_data = [[1, 0, 1]]
test_set = np.array(test_data)


def q1knn(newInput, dataSet, labels, k):
    result = []
    distances = []
    voters = []
    for i in dataSet:
        print(i)
        distances.append(np.linalg.norm(i - newInput))
    kmin_index = np.argpartition(distances, k)
    print(kmin_index)
    print(kmin_index[:k])
    for i in kmin_index[:k]:
        voters.append(labels[i])
    print(voters)
    [(predict_label, appear_time)] = Counter(voters).most_common(1)
    result.append(predict_label)
    return result


print(q1knn(test_set, train_set, train_label, 3))

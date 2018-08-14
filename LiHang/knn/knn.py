import numpy as np
import pandas as pd
import cv2
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_hog_features(trainset):
    features = []
    hog = cv2.HOGDescriptor('../hog.xml')
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))
    return features
def Predict(testset,trainset,train_labels):
    predict = []
    count = 0

    for test_vec in testset:
        print(count)
        count += 1

        knn_list = []
        max_index = -1
        max_distance = 0

        for i in range(k):
            label = train_labels[i]
            trainset_vec = trainset[i]

            distance = np.linalg.norm(trainset_vec - test_vec)
            knn_list.append((distance,label))

        for i in range(k,len(train_labels)):
            labels = train_labels[i]
            trainset_vec = trainset[i]
            distance = np.linalg.norm(trainset_vec - test_vec)

            if max_index < 0:
                for j in range(k):
                    if max_distance < knn_list[j][0]:
                        max_index = j
                        max_distance = knn_list[j][0]
            if distance < max_distance:
                knn_list[max_index] = (distance,label)
                max_index = -1
                max_distance = 0

        class_total = 10
        class_count = [0 for i in range(class_total)]
        for distance,label in knn_list:
            class_count[label] += 1

        max_vote = max(class_count)
        for i in range(class_total):
            if class_count[i] == max_vote:
                predict.append(i)
                break
    return np.array(predict)

k = 10

if __name__ == "__main__":

    print('start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values
    imgs = data[:,1:]
    labels = data[:,0]
    features = get_hog_features(imgs)
    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=23323)
    time_2 = time.time()
    print('read data cost',time_2 - time_1,'second')

    print('start training')
    print('knn do not need to train')
    time_3 = time.time()
    print('training cost',time_3 - time_2,'second')

    print('start predicting')
    test_pridict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print('predicting cost',time_4 - time_3,'second')

    score = accuracy_score(test_labels,test_pridict)
    print('the accuracy score is',score)
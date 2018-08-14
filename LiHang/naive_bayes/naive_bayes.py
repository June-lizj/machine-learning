import numpy as np
import pandas as pd
import cv2
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
    return cv_img

def Train(trainset,train_labels):
    prior_probability = np.zeros(class_num)
    conditional_probability = np.zeros((class_num,feature_len,2))

    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])
        label = train_labels[i]

        prior_probability[label] += 1

        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1

    for i in range(class_num):
        for j in range(feature_len):

            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            probability_0 = (float(pix_0) / float(pix_0 + pix_1)) * 1000000 + 1
            probability_1 = (float(pix_1) / float(pix_0 + pix_1)) * 1000000 + 1

            conditional_probability[i][j][0] = probability_0
            conditional_probability[i][j][1] = probability_1

    return prior_probability,conditional_probability

def calculate_probability(img,label):
    probabity = int(prior_probability[label])

    for j in range(len(img)):
        probabity *= int(conditional_probability[label][j][img[j]])

    return probabity

def Predict(testset,prior_probability,conditional_probability):
    predict = []

    for img in testset:

        img = binaryzation(img)

        max_label = 0
        max_probability = calculate_probability(img,0)

        for j in range(1,10):
            probability = calculate_probability(img,j)

            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)

    return np.array(predict)

class_num = 10
feature_len = 784

if __name__ == "__main__":

    print("start read data")
    time_1 = time.time()
    raw_data = pd.read_csv("../data/train.csv")
    data = raw_data.values
    img = data[:,1:]
    labels = data[:,0]
    train_features,test_features,train_labels,test_labels = train_test_split(img,labels,test_size=0.3,random_state=23323)
    time_2 = time.time()
    print("read data cost",time_2 - time_1,"seconds")

    print("start training")
    prior_probability,conditional_probability = Train(train_features,train_labels)
    time_3 = time.time()
    print("training cost",time_3 - time_2,"seconds")

    print("start predicting")
    test_predict = Predict(test_features,prior_probability,conditional_probability)
    time_4 = time.time()
    print("predict cost",time_4 - time_3,"seconds")

    score = accuracy_score(test_labels,test_predict)
    print("the accuracy score is",score)

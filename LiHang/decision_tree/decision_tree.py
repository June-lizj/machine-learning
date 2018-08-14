import numpy as np
import pandas as pd
import time
import cv2
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10

def log(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args,**kwargs)

        end_time = time.time()
        logging.debug('end %s(),cost %s seconds' % (func.__name__,end_time - start_time))
        return ret
    return wrapper

def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
    return  cv_img

@log
def binaryzation_features(trainset):
    features = []
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        cv_img_2 = binaryzation(cv_img)
        features.append(cv_img_2)
    features = np.array(features)
    features = np.reshape(features,(-1,784))

class Tree(object):
    def __init__(self,node_type,Class = None,feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature
    def add_tree(self,val,tree):
        self.dict[val] = tree
    def predict(self,features):
        if(self.node_type == 'leaf'):
            return self.Class
        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def calculate_entropy(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    entropy = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        log_p = np.log2(p)
        entropy -= p * log_p
    return entropy

def calculate_conditional_entropy(x,y):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    entropy = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        tmp_entropy = calculate_entropy(sub_y)
        entropy += (float(sub_y.shape[0]) / y.shape[0]) * tmp_entropy
    return entropy

def calculate_entropy_grap(x,y):
    #计算信息增益
    base_entropy = calculate_entropy(x)
    conditional_entropy = calculate_conditional_entropy(x,y)
    entropy_grap = base_entropy - conditional_entropy
    return entropy_grap

def recurse_train(train_set,train_label,features,epsilon):
    global total_class
    LEAF = 'leaf'
    INTERNAL = 'internal'

    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF,Class = label_set.pop())

    (max_class,max_len) = max([(i,len(filter(lambda x:x==i,train_label))) for i in range(total_class)],key=lambda x:x[1])

    if len(features) == 0:
        return Tree(LEAF,Class = max_class)

    max_feature = 0
    max_gda = 0

    D = train_label
    HD = calculate_entropy(D)
    for feature in features:
        A = np.array(train_set[:,feature].flat)
        gda = HD - calculate_conditional_entropy(A,D)
        if gda > max_gda:
            max_gda,max_feature = gda,feature

    if max_gda < epsilon:
        return Tree(LEAF,Class=max_class)

    sub_features = filter(lambda x:x!=max_feature,features)
    tree = Tree(INTERNAL,Class=max_feature)

    feature_col = np.array(train_set[:,max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:
        index = []
        for i in len(train_label):
            if train_set[i][max_feature] == feature_value:
                index.append(i)
        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set,sub_train_label,sub_features,epsilon)
        tree.add_tree(feature_value,sub_tree)

    return tree

@log
def train(train_set,train_label,features,epsilon):
    return recurse_train(train_set,train_label,features,epsilon)

@log
def predict(test_set,tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    raw_data = pd.read_csv("../data/train.csv",header=0)
    data = raw_data.values

    imgs = data[:,1:]
    labels = data[:,0]

    features = binaryzation_features(imgs)

    train_features,test_features,train_labels,test_labels = train_test_split(imgs,labels,test_size=0.3,random_state=23323)

    tree = train(train_features,train_labels,[i for i in range(784)],0.1)
    test_predict = predict(test_features,tree)
    score = accuracy_score(test_labels,test_predict)

    print('the accuracy score is',score)



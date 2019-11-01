#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


# In[6]:


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


# In[7]:


def compute_dsift(img, stride, size):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()

    step_size = 20
    
    ## 不确定 KeyPoint(x, y, size)
    kp = [cv2.KeyPoint(x+size/2, y+size/2, size) for y in range(0, img.shape[0], stride) 
                                    for x in range(0, img.shape[1], stride)]

    kp1, dense_feature = sift.compute(img , kp)
    return dense_feature


# In[8]:


def get_tiny_image(img, output_size):
    # To do
    img_resize = cv2.resize(img, output_size, interpolation = cv2.INTER_AREA)
    #normalization
    
    feature = np.zeros(output_size)
    
    feature_omean= img_resize- np.mean(img_resize)
       
    feature = feature_omean / np.linalg.norm(feature_omean.reshape((1,np.prod(feature_omean.shape))))
    
    return feature


# In[25]:


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    nbrs = NearestNeighbors(algorithm='auto').fit(feature_train)
    
    distances_map,indices_map =nbrs.kneighbors(feature_test,n_neighbors=k)
    
    label_test_pred = np.zeros(len(indices_map))
    
    for num in range(len(indices_map)):
        label_test_pred[num] = np.argmax(np.bincount(label_train[indices_map[num,:]]))
    
    return label_test_pred


# In[26]:


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    #1
    feature_vec = []
    output_size = output_size= (16,16)
    k = 10
    for name in img_train_list :
        img = cv2.imread(name,0)
        feature = get_tiny_image(img, output_size)
        feature_vec.append( feature.reshape(np.prod(feature.shape)))
    #2
    feature_test_vec = []
    for name in img_test_list :
        img = cv2.imread(name,0)
        feature_test = get_tiny_image(img, output_size)
        feature_test_vec.append( feature_test.reshape(np.prod(feature_test.shape)))  
    #3  
    label_train_set = []
    for item in label_train_list:
        label_train_set.append(label_classes.index(item))
    
    label_train_set = np.array(label_train_set)
    #4
    label_test_set = []
    for item in label_test_list:
        label_test_set.append(label_classes.index(item))
    
    label_test_set = np.array(label_test_set)
    
    
    #predict
    label_test_pred = predict_knn(feature_vec, label_train_set, feature_test_vec, k)
    
    #confusion matrix]
    from sklearn.metrics import confusion_matrix
    
    y_true = label_test_set
    y_pred = label_test_pred
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(confusion)/np.sum(np.sum(confusion))
    
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# In[28]:


def build_visual_dictionary(dense_feature_list, dic_size=50):
    # To do
    dense_feature=np.zeros((1,128))
    
    for item in dense_feature_list:

        dense_feature = np.concatenate((dense_feature, item), axis=0)
    
    dense_feature_set = np.delete(dense_feature, 0, 0)
    
    kmeans = KMeans(n_clusters = 50,n_init=10,max_iter=300).fit(dense_feature_set)
    vocab = kmeans.cluster_centers_
    return vocab


# In[29]:


def compute_bow(feature, vocab):
    # To do
    nbrs = NearestNeighbors(algorithm='auto').fit(vocab) 
    
    distances_map,indices_map =nbrs.kneighbors(feature,n_neighbors=1)
    
    bow_feature_pre = indices_map.reshape((1,len(vocab)))
    
    bow_feature = np.bincount(bow_feature_pre,minlength=len(vocab))/len(vocab)
    
    return bow_feature


# In[30]:


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride, size = (20,20)
    
    dense_feature_list=[]
    for name in img_train_list :
        img = cv2.imread(name,0)
        stride, size = (20,20)
        dense_feature = compute_dsift(img, stride, size)
        dense_feature_list.append(dense_feature)

    vocab = build_visual_dictionary(dense_feature_list, 50)
    np.savetxt('test_knn_bow_out', vocab, delimiter=',')
    
    vocab_feature_list=[]
    for name in img_train_list :
        img = cv2.imread(name,0)
        stride, size = (20,20)
        dense_feature = compute_dsift(img, stride, size)
        
        bow_feature = compute_bow(dense_feature, vocab)
        
        vocab_feature_list.append(bow_feature)
    
    #2
    vocab_feature_test_list = []
    for name in img_test_list :
        img = cv2.imread(name,0)
        dense_feature_test = compute_dsift(img, stride, size)
        
        bow_feature_test = compute_bow(dense_feature_test, vocab)
        
        vocab_feature_test_list.append(dense_feature_test)
        
    #3  
    label_train_set = []
    for item in label_train_list:
        label_train_set.append(label_classes.index(item))
    
    label_train_set = np.array(label_train_set)
    #4
    label_test_set = []
    for item in label_test_list:
        label_test_set.append(label_classes.index(item))
    
    label_test_set = np.array(label_test_set)
    
    k = 10
    
    label_test_pred = predict_knn(vocab_feature_list, label_train_set, vocab_feature_test_list, k)
    
    
    from sklearn.metrics import confusion_matrix
    
    y_true = label_test_set
    y_pred = label_test_pred
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(confusion)/np.sum(np.sum(confusion))
    
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


# In[24]:


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    clf = LinearSVC(tol=1e-5,C=1.0) 
    clf.fit(feature_train, label_train)  
    
    label_test_pred = clf.predict(feature_test)
    
    
    return label_test_pred


# In[ ]:


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    
    dense_feature_list=[]
    for name in img_train_list :
        img = cv2.imread(name,0)
        stride, size = (20,20)
        dense_feature = compute_dsift(img, stride, size)
        dense_feature_list.append(dense_feature)

    vocab = build_visual_dictionary(dense_feature_list, 50)
    np.savetxt('test_svm_bow_out', vocab, delimiter=',')
    
    vocab_feature_list=[]
    for name in img_train_list :
        img = cv2.imread(name,0)
        stride, size = (20,20)
        dense_feature = compute_dsift(img, stride, size)
        
        bow_feature = compute_bow(dense_feature, vocab)
        
        vocab_feature_list.append(bow_feature)
    
    #2
    vocab_feature_test_list = []
    for name in img_test_list :
        img = cv2.imread(name,0)
        dense_feature_test = compute_dsift(img, stride, size)
        
        bow_feature_test = compute_bow(dense_feature_test, vocab)
        
        vocab_feature_test_list.append(dense_feature_test)
        
    #3  
    label_train_set = []
    for item in label_train_list:
        label_train_set.append(label_classes.index(item))
    
    label_train_set = np.array(label_train_set)
    #4
    label_test_set = []
    for item in label_test_list:
        label_test_set.append(label_classes.index(item))
    
    label_test_set = np.array(label_test_set)
    
    label_test_pred = predict_svm(feature_train, label_train, feature_test, n_classes)
    
    from sklearn.metrics import confusion_matrix
    
    y_true = label_test_set
    y_pred = label_test_pred
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(confusion)/np.sum(np.sum(confusion))
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# In[ ]:


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


# In[ ]:


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)


# In[ ]:





# assorted utils for QKernel experiments

import os
import gzip
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from functools import reduce

import tensorflow as tf


utils_folder = Path(__file__).parent


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`
    Train: kind='train'
    Test: kind='t10k'
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_random_dataset(dataset_dim, n_train, n_test): # generate random training data
    
    x_train = np.random.uniform(0, 2*np.pi, (n_train,dataset_dim))
    y_train = np.random.choice([-1,1], n_train, p=[0.5,0.5])
    
    x_test = np.random.uniform(0, 2*np.pi, (n_test,dataset_dim))
    y_test = np.random.choice([-1,1], n_test, p=[0.5,0.5])
    return x_train, x_test, y_train, y_test


def get_mnist_dataset(dataset_dim,n_train,n_test):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)
    
    def filter_01(x, y):
        keep = (y == 0) | (y == 1)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y
    
    x_train, y_train = filter_01(x_train, y_train)
    x_test, y_test = filter_01(x_test, y_test)
    
    #normalze
    x_train, x_test = x_train/255.0, x_test/255.0
    feature_mean = np.mean(x_train,axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean
    
    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    
    count_train0 = 0
    count_train1 = 0
    
    count_test0 = 0
    count_test1 = 0
    
    num_half_train = int(n_train/2)
    num_half_test = int(n_test/2)
    
    x_train_final = np.zeros((n_train,dataset_dim))
    x_test_final = np.zeros((n_test,dataset_dim))
    

    for i in range(len(y_train)):
        if y_train[i] == True:
            if count_train0 < num_half_train:
                x_train_final[count_train0] = x_train[i]
                count_train0 += 1
            
        else:
            if count_train1 < num_half_train:
                x_train_final[num_half_train + count_train1] = x_train[i]
                count_train1 += 1
        
        if count_train0 + count_train1 == n_train:
            break
    
    for i in range(len(y_test)):
        if y_test[i] == True:
            if count_test0 < num_half_test:
                x_test_final[count_test0] = x_test[i]
                count_test0 += 1
            
        else:
            if count_test1 < num_half_test:
                x_test_final[num_half_test + count_test1] = x_test[i]
                count_test1 += 1
        
        if count_test0 + count_test1 == n_test:
            break
    
    y_train_final0 = np.zeros(num_half_train) - 1
    y_train_final1 = np.zeros(num_half_train) + 1
    y_train_final = np.concatenate((y_train_final0 ,y_train_final1))
    
    y_test_final0 = np.zeros(num_half_test) - 1
    y_test_final1 = np.zeros(num_half_test) + 1
    y_test_final = np.concatenate((y_test_final0 ,y_test_final1))
    
    return x_train_final, x_test_final, y_train_final, y_test_final


def get_fashion_mnist_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://github.com/zalandoresearch/fashion-mnist
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    path = Path(utils_folder, '../data/fashion_mnist/')
    x_train, y_train = load_mnist(path, kind='train')
    x_test, y_test = load_mnist(path, kind='t10k')
    def filter_03(x, y):
        keep = (y == 0) | (y == 3)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y
    
    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)

    # normalize
    x_train, x_test = x_train/255.0, x_test/255.0
    feature_mean = np.mean(x_train,axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)

    count_train0 = 0
    count_train1 = 0
    
    count_test0 = 0
    count_test1 = 0
    
    num_half_train = int(n_train/2)
    num_half_test = int(n_test/2)
    
    x_train_final = np.zeros((n_train,dataset_dim))
    x_test_final = np.zeros((n_test,dataset_dim))
    

    for i in range(len(y_train)):
        if y_train[i] == True:
            if count_train0 < num_half_train:
                x_train_final[count_train0] = x_train[i]
                count_train0 += 1
            
        else:
            if count_train1 < num_half_train:
                x_train_final[num_half_train + count_train1] = x_train[i]
                count_train1 += 1
        
        if count_train0 + count_train1 == n_train:
            break
    
    for i in range(len(y_test)):
        if y_test[i] == True:
            if count_test0 < num_half_test:
                x_test_final[count_test0] = x_test[i]
                count_test0 += 1
            
        else:
            if count_test1 < num_half_test:
                x_test_final[num_half_test + count_test1] = x_test[i]
                count_test1 += 1
        
        if count_test0 + count_test1 == n_test:
            break
    
    y_train_final0 = np.zeros(num_half_train) - 1
    y_train_final1 = np.zeros(num_half_train) + 1
    y_train_final = np.concatenate((y_train_final0 ,y_train_final1))
    
    y_test_final0 = np.zeros(num_half_test) - 1
    y_test_final1 = np.zeros(num_half_test) + 1
    y_test_final = np.concatenate((y_test_final0 ,y_test_final1))
    
    return x_train_final, x_test_final, y_train_final, y_test_final


def get_kuzushiji_mnist_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://github.com/rois-codh/kmnist
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    x_train = np.load(Path(utils_folder, '../data/kmnist/kmnist-train-imgs.npz'))['arr_0'].reshape(60000,784)
    y_train = np.load(Path(utils_folder, '../data/kmnist/kmnist-train-labels.npz'))['arr_0']
    x_test = np.load(Path(utils_folder, '../data/kmnist/kmnist-test-imgs.npz'))['arr_0'].reshape(10000,784)
    y_test = np.load(Path(utils_folder, '../data/kmnist/kmnist-test-labels.npz'))['arr_0']
    
    def filter_14(x, y):
        keep = (y == 1) | (y == 4)
        x, y = x[keep], y[keep]
        y = y == 1
        return x,y
    
    x_train, y_train = filter_14(x_train, y_train)
    x_test, y_test = filter_14(x_test, y_test)

    # normalize
    x_train, x_test = x_train/255.0, x_test/255.0
    feature_mean = np.mean(x_train, axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    
    count_train0 = 0
    count_train1 = 0
    
    count_test0 = 0
    count_test1 = 0
    
    num_half_train = int(n_train/2)
    num_half_test = int(n_test/2)
    
    x_train_final = np.zeros((n_train,dataset_dim))
    x_test_final = np.zeros((n_test,dataset_dim))
    

    for i in range(len(y_train)):
        if y_train[i] == True:
            if count_train0 < num_half_train:
                x_train_final[count_train0] = x_train[i]
                count_train0 += 1
            
        else:
            if count_train1 < num_half_train:
                x_train_final[num_half_train + count_train1] = x_train[i]
                count_train1 += 1
        
        if count_train0 + count_train1 == n_train:
            break
    
    for i in range(len(y_test)):
        if y_test[i] == True:
            if count_test0 < num_half_test:
                x_test_final[count_test0] = x_test[i]
                count_test0 += 1
            
        else:
            if count_test1 < num_half_test:
                x_test_final[num_half_test + count_test1] = x_test[i]
                count_test1 += 1
        
        if count_test0 + count_test1 == n_test:
            break
    
    y_train_final0 = np.zeros(num_half_train) - 1
    y_train_final1 = np.zeros(num_half_train) + 1
    y_train_final = np.concatenate((y_train_final0 ,y_train_final1))
    
    y_test_final0 = np.zeros(num_half_test) - 1
    y_test_final1 = np.zeros(num_half_test) + 1
    y_test_final = np.concatenate((y_test_final0 ,y_test_final1))
    
    return x_train_final, x_test_final, y_train_final, y_test_final


def get_plasticc_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://arxiv.org/abs/2101.09581
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    data = np.load(open(Path(utils_folder, '../data/plasticc_data/SN_67floats_preprocessed.npy'), 'rb'))

    X = data[:,:67]
    Y = data[:,67]
    
    x_train_normalized, x_test_normalized, y_train, y_test = train_test_split(X, Y, train_size=n_train, test_size=n_test, random_state=42, stratify=Y)
    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    
    
    return x_train, x_test, y_train, y_test


def get_dataset(name, dataset_dim, n_train, n_test):
    if name == 'fashion-mnist':
        return get_fashion_mnist_dataset(dataset_dim, n_train, n_test)
    elif name == 'kmnist':
        return get_kuzushiji_mnist_dataset(dataset_dim, n_train, n_test)
    elif name == 'plasticc':
        return get_plasticc_dataset(dataset_dim, n_train, n_test)
    elif name == 'random':
        return get_random_dataset(dataset_dim, n_train, n_test)
    elif name == 'mnist':
        return get_mnist_dataset(dataset_dim, n_train, n_test)
    else:
        raise ValueError(f"Unknown dataset: {name}")



###############################################################################


def get_mnist_resize(dataset_dim, n_train, n_test):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    
    
    def filter_01(x, y):
        keep = (y == 0) | (y == 1)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y
    
    x_train, y_train = filter_01(x_train, y_train)
    x_test, y_test = filter_01(x_test, y_test)
    
    x_train = tf.image.resize(x_train, (dataset_dim,1)).numpy()
    x_test = tf.image.resize(x_test, (dataset_dim,1)).numpy()
    
    
    x_train = x_train.reshape(12665,dataset_dim)
    x_test = x_test.reshape(2115,dataset_dim)
    
    ##############################
    
    count_train0 = 0
    count_train1 = 0
    
    count_test0 = 0
    count_test1 = 0
    
    num_half_train = int(n_train/2)
    num_half_test = int(n_test/2)
    
    x_train_final = np.zeros((n_train,dataset_dim))
    x_test_final = np.zeros((n_test,dataset_dim))
    

    for i in range(len(y_train)):
        if y_train[i] == True:
            if count_train0 < num_half_train:
                x_train_final[count_train0] = x_train[i]
                count_train0 += 1
            
        else:
            if count_train1 < num_half_train:
                x_train_final[num_half_train + count_train1] = x_train[i]
                count_train1 += 1
        
        if count_train0 + count_train1 == n_train:
            break
    
    for i in range(len(y_test)):
        if y_test[i] == True:
            if count_test0 < num_half_test:
                x_test_final[count_test0] = x_test[i]
                count_test0 += 1
            
        else:
            if count_test1 < num_half_test:
                x_test_final[num_half_test + count_test1] = x_test[i]
                count_test1 += 1
        
        if count_test0 + count_test1 == n_test:
            break
    
    y_train_final0 = np.zeros(num_half_train) - 1
    y_train_final1 = np.zeros(num_half_train) + 1
    y_train_final = np.concatenate((y_train_final0 ,y_train_final1))
    
    y_test_final0 = np.zeros(num_half_test) - 1
    y_test_final1 = np.zeros(num_half_test) + 1
    y_test_final = np.concatenate((y_test_final0 ,y_test_final1))
    
    return x_train_final, x_test_final, y_train_final, y_test_final
    



















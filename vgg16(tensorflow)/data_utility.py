# -*- coding:utf-8 -*-

import os
import sys
import time
import pickle
import random
import math
import numpy as np
import pandas as pd

class_num = 10
image_size = 32
img_channels = 3


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #


def download_data(dataset_name = 'cifar10'):
    dirname = ''
    origin = ''
    fname = ''
    if dataset_name == 'cifar100':
        dirname = 'cifar-100-batches-py'
        origin = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        fname = 'cifar-100-python.tar.gz'
    elif dataset_name == 'cifar10':
        dirname = 'cifar-10-batches-py'
        origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        fname = 'cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet aready exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            if duration == 0:
                duration = 0.1
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration + 0.00001))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()
        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    if class_num == 10:
        labels = batch[b'labels']
    else:
        labels = batch[b'fine_labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def prepare_data(dataset_name):
    print("======Loading data======")
    download_data(dataset_name)
    image_dim = image_size * image_size * img_channels
    train_data = 0
    train_labels = 0
    test_data = 0
    test_labels = 0
    label_names = []
    global class_num
    if dataset_name == 'cifar10':
        class_num = 10
        data_dir = './cifar-10-batches-py'
        meta = unpickle(data_dir + '/batches.meta')

        label_names = meta[b'label_names']
        label_count = len(label_names)
        train_files = ['data_batch_%d' % d for d in range(1, 6)]
        train_data, train_labels = load_data(train_files, data_dir, label_count)
        test_data, test_labels = load_data(['test_batch'], data_dir, label_count)
    elif dataset_name == 'cifar100':
        class_num = 100
        data_dir = './cifar-100-python'
        meta = unpickle(data_dir + '/meta')
        label_names = meta[b'fine_label_names']
        label_count = len(label_names)
        train_files = ['train']
        train_data, train_labels = load_data(train_files, data_dir, label_count)
        test_data, test_labels = load_data(['test'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))#随机打乱
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels, label_names

# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# ├─ data_preprocessing()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch

def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test

'''train_x, train_y, test_x, test_y, l_label_name= prepare_data('cifar100')
d_cifar10 = {}
d_cifar100 = {}
cifar10 = pd.read_csv('./cifar10vec.csv', header=0, encoding='utf8')
cifar100 = pd.read_csv('./cifar100vec.csv', header=0, encoding='utf8')
#print(cifar100)
print(l_label_name)
num_dic = {}
file = open("key_value", "w")
for i in range(100):
    num_dic[l_label_name[i].decode()] = i
    file.write(l_label_name[i].decode() + "\n")
file.close()
print(num_dic)

print(len(num_dic))
for i in range(100):
    cifar100.iloc[i, 0] = num_dic[cifar100.iloc[i, 0]]
#print(cifar100)
cifar100.to_csv("cifar100_numindex_vec.csv", index = False)'''


#print(train_label)
#print(np.argmax(train_y, 1))
#print(l_label_name)
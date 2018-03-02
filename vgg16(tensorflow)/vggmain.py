# -*- coding:utf-8 -*-
# ========================================================== #
# File name: vgg_19.py
# Author: BIGBALLON
# Date created: 07/22/2017
# Python Version: 3.5.2
# Description: implement vgg19 network to train cifar10
# Result: test accuracy about 93.28% - 93.32%
# ========================================================== #

import tensorflow as tf
import VGG
from data_utility import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

iterations = 200
batch_size = 250
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_logs'
model_save_path = './testmodel/model.ckpt'
model_restore_path = './testmodel/'#'./norelumodel/'
dataset_name = 'cifar100'
class_num = 100
d_cifar10 = {}
d_cifar100 = {}
if dataset_name == 'cifar100':
    class_num = 100

# ========================================================== #
# ├─ cal_d_cifar()
# ├─ run_testing()
# └─ learning_rate_schedule()
# ========================================================== #
def cal_d_cifar():#计算d_cifar
    global  d_cifar100
    global  d_cifar10
    cifar10 = pd.read_csv('./cifar10vec.csv', header=0, encoding='utf8')
    cifar100 = pd.read_csv('./cifar100vec.csv', header=0, encoding='utf8')
    for i in range(10):
        d_cifar10[cifar10.iloc[i, 0]] = (np.array(cifar10.iloc[i, 1:301]) + 1)/2
    for i in range(100):
        d_cifar100[cifar100.iloc[i, 0]] = (np.array(cifar100.iloc[i, 1:301]) + 1)/2

def getcifar_veclabel(train_y):
    n, m = np.shape(train_y)
    l_name = [l_label_name[int(np.argmax(train_y[i]))].decode() for i in range(n)]
    train_label = np.zeros((n, 300))
    flag = False
    for i in range(n):
        if dataset_name == 'cifar100':
            train_label[i, :] = d_cifar100[l_name[i]]
        elif dataset_name == 'cifar10':
            train_label[i, :] = d_cifar10[l_name[i]]
        else:
            print(flag)
    return train_label

# ========================================================== #
# ├─ main()
# Training and Testing
# Save train/teset loss and acc for visualization
# Save Model in ./model
# ========================================================== #

def pretrain(is_relu, lr = 0.05, epoch = 164, seed = 5):
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, 300])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    # build_network
    output = VGG.vgg19(x, keep_prob, train_flag, class_num=300, onemore_layer=False, stop_flag=False, is_relu=is_relu, seed = seed)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, output), 1)) 
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(
        cross_entropy + l2 * weight_decay)
    saver = tf.train.Saver()

    def learning_rate_schedule(epoch_num):
        if epoch_num < 81:
            return lr
        elif epoch_num < 121:
            return lr / 5
        else:
            return lr / 10

    def run_testingloss(sess):
        loss = 0.0
        pre_index = 0
        add = 250
        iter = 40
        for it in range(iter):
            batch_x = test_x[pre_index:pre_index + add]
            batch_y = test_y[pre_index:pre_index + add]
            batch_y = getcifar_veclabel(batch_y)
            pre_index = pre_index + add
            loss_ = sess.run(cross_entropy,
                             feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
            loss += loss_ / iter
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss)])
        return loss, summary

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)
        # make sure [bath_size * iteration = data_set_number]
        for ep in range(1, epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_loss = 0.0
            start_time = time.time()

            print("\nepoch %d/%d:" % (ep, epoch))
            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)
                batch_y = getcifar_veclabel(batch_y)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate, learning_rate: lr,
                                                    train_flag: True})

                train_loss += batch_loss
                pre_index += batch_size
                if it == iterations:
                    train_loss /= iterations
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)])

                    val_loss, test_summary = run_testingloss(sess)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print(
                        "iteration: %d/%d, cost_time: %ds, train_loss: %.4f, test_loss: %.4f" % (
                            it, iterations, int(time.time() - start_time), train_loss, val_loss))
                else:
                    print("iteration: %d/%d, train_loss: %.4f" % (
                    it, iterations, train_loss / it), end='\r')
        model_save_path = "./vec_pretrain_%depoch_model/model.ckpt" % epoch
        save_path = saver.save(sess, model_save_path)
        print("Model saved in file: %s" % save_path)

def normaltrain(restore_flag, save_flag, onemore_layer, stop_flag, is_relu, lr = 0.05, epoch = 164, restore_epoch = 10, seed = 5, restore_all = False):
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    # build_network
    output = VGG.vgg19(x, keep_prob, train_flag, class_num, onemore_layer, stop_flag, is_relu, seed = seed)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(
        cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    restorelist = []
    all_vars = tf.trainable_variables()
    print(all_vars)
    if not restore_all:
        for v in all_vars:
            if not v.name in ["fc3:0", "Variable_18:0", "BatchNorm_18/beta:0", "BatchNorm_18/gamma:0"]:
                restorelist.append(v)
    else:
        for v in all_vars:
            restorelist.append(v)
    # initial an saver to save model
    saver = tf.train.Saver(restorelist)
    
    def learning_rate_schedule(epoch_num):
        if epoch_num < 81:
            return lr
        elif epoch_num < 121:
            return lr / 5
        else:
            return lr / 10

    def run_testing(sess, ep):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = batch_size
        iter = int(10000 / batch_size)
        for it in range(iter):
            batch_x = test_x[pre_index:pre_index + add]
            batch_y = test_y[pre_index:pre_index + add]
            pre_index = pre_index + add
            loss_, acc_ = sess.run([cross_entropy, accuracy],
                                   feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
            loss += loss_ / iter
            acc += acc_ / iter
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                    tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
        return acc, loss, summary

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)
        if restore_flag:
            model_restore_path = "./vec_pretrain_%depoch_model/model.ckpt" % restore_epoch
            #model_restore_path = "./testmodel/model.ckpt"
            saver.restore(sess, model_restore_path)
        for ep in range(1, epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\nepoch %d/%d:" % (ep, epoch))
            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate, learning_rate: lr,
                                                    train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size
                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print(
                        "iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" % (
                            it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                    it, iterations, train_loss / it, train_acc / it), end='\r')

        if save_flag:
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_save_path)
            print("Model saved in file: %s" % save_path)

if __name__ == '__main__':

    train_x, train_y, test_x, test_y, l_label_name= prepare_data(dataset_name)
    train_x, test_x = data_preprocessing(train_x, test_x)
    cal_d_cifar()
    #define placeholder x, y_ , keep_prob, learning_rate
    #class_num = 100
    #pretrain(is_relu=False, lr = 0.05, epoch = 200)
    normaltrain(restore_flag=False, save_flag=True, onemore_layer=False, stop_flag=False, is_relu=False, lr = 0.05, epoch = 200, restore_epoch = 300, seed = 5)


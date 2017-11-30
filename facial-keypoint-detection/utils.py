from datetime import datetime
import os
import time

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import tensorflow as tf


def load(test=False, cols=None, verbose=False):
    FTRAIN = os.environ["FTRAIN"]
    FTEST = os.environ["FTEST"]

    fname = FTEST if test else FTRAIN
    df = read_csv(fname)

    df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ["Image"]]
    if verbose:
        print(df.count())

    df = df.dropna()

    X = np.vstack(df["Image"].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def plot_sample(x, y, axis):
    # 테스트 이미지에 keypoint 를 그려보는것도 블로그 포스트의 코드를 그대로 사용했다.
    #  크기를 키우고 (figsize), marker 크기와(s), 색깔에(c) 변화를 줬다.
    img = x.reshape(96, 96)
    axis.imshow(img, cmap="gray")
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=50, c='r')


def train(num_epoch, X, y, X_test, name="", ratio_val=0.2):
    """
    https://www.tensorflow.org/get_started/mnist/mechanics
    tensorflow feed forward example 로 검색해서 나온 [코드를](https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0) 참고해봤다.
    mini-batch training 에 필요한 hyper-parameter 들은 [Kaggle 이 소개한 블로그 포스트를](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) 따라해보자.
    위의 블로그 포스트에서도 첫번째 model 에서는 full-batch 로 학습시키는것 같으므로, 우선은 mini-batch 는 구현하지 않고 진행해보자.
    """
    # notebook 에서 train 함수를 호출할때 InvalidArgumentError 가 발생하는걸 방지하기 위함
    tf.reset_default_graph()

    # X 가 매번 shuffle 된다고 가정하고 연속된 2 덩어리로 나눕니다.
    num_total, _ = X.shape
    num_val = int(num_total * ratio_val)
    X_train = X[:num_val, :]
    y_train = y[:num_val, :]
    X_val = X[num_val:, :]
    y_val = y[num_val:, :]

    images = tf.placeholder(tf.float32, shape=(None, 9216))
    truth = tf.placeholder(tf.float32, shape=(None, 30))

    with tf.name_scope("hidden"):
        w1 = tf.get_variable("W1", shape=[9216, 100], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.zeros([100]), name="biases")
        hidden = tf.nn.relu(tf.matmul(images, w1) + b1)
    with tf.name_scope("mse_linear"):
        w2 = tf.get_variable("W2", shape=[100, 30], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.zeros([30]), name="biases")
        predict = tf.matmul(hidden, w2) + b2

    loss = tf.losses.mean_squared_error(truth, predict)
    # MNIST 쪽 예제에서는 global_step 을 만들어서 추가로 넣어줬었다.
    # global_step = tf.Variable(0, name="global_step", trainable=False)
    # train_op = optimizer.minimize(loss, global_step=global_step)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_summary = tf.summary.scalar("loss", loss)

        datetime_now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        train_summary_path = "{}_{}_train".format(datetime_now, name)
        val_summary_path = "{}_{}_val".format(datetime_now, name)
        train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", train_summary_path))
        val_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", val_summary_path))

        feed_dict_train = {images: X_train, truth: y_train}
        feed_dict_val = {images: X_val, truth: y_val}

        for e in range(num_epoch):
            start_ms = int(time.time() * 1000)
            sess.run(optimizer, feed_dict=feed_dict_train)
            train_loss_summary, train_loss = sess.run([loss_summary, loss], feed_dict=feed_dict_train)
            val_loss_summary = sess.run(loss_summary, feed_dict=feed_dict_val)
            elapsed_ms = int(time.time() * 1000) - start_ms
            print("Epoch: {:4d}\tTraining Loss: {:.7f}\tElapsed Time(ms): {:6d}".format(e, train_loss, elapsed_ms))
            train_writer.add_summary(train_loss_summary, global_step=e)
            val_writer.add_summary(val_loss_summary, global_step=e)
            train_writer.flush()
            val_writer.flush()

        y_pred = sess.run(predict, feed_dict={images: X_test})

    return y_pred


def train_cnn(num_epoch, X, y, X_test, name, ratio_val=0.2):
    tf.reset_default_graph()

    num_total, _ = X.shape
    num_val = int(num_total * ratio_val)
    X_train = X[:num_val, :]
    y_train = y[:num_val, :]
    X_val = X[num_val:, :]
    y_val = y[num_val:, :]

    images = tf.placeholder(tf.float32, shape=(None, 9216))
    truth = tf.placeholder(tf.float32, shape=(None, 30))

    input_layer = tf.reshape(images, [-1, 96, 96, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 12 * 12 * 128])
    dense1 = tf.layers.dense(inputs=pool3_flat, units=500, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=500, activation=tf.nn.relu)
    predict = tf.layers.dense(inputs=dense2, units=30, activation=None)

    loss = tf.losses.mean_squared_error(truth, predict)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_summary = tf.summary.scalar("loss", loss)

        datetime_now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        train_summary_path = "{}_{}_train".format(datetime_now, name)
        val_summary_path = "{}_{}_val".format(datetime_now, name)
        train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", train_summary_path))
        val_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", val_summary_path))

        feed_dict_train = {images: X_train, truth: y_train}
        feed_dict_val = {images: X_val, truth: y_val}

        for e in range(num_epoch):
            start_ms = int(time.time() * 1000)
            sess.run(optimizer, feed_dict=feed_dict_train)
            train_loss_summary, train_loss = sess.run([loss_summary, loss], feed_dict=feed_dict_train)
            val_loss_summary = sess.run(loss_summary, feed_dict=feed_dict_val)
            elapsed_ms = int(time.time() * 1000) - start_ms
            print("Epoch: {:4d}\tTraining Loss: {:.7f}\tElapsed Time(ms): {:6d}".format(e, train_loss, elapsed_ms))
            train_writer.add_summary(train_loss_summary, global_step=e)
            val_writer.add_summary(val_loss_summary, global_step=e)
            train_writer.flush()
            val_writer.flush()

        y_pred = sess.run(predict, feed_dict={images: X_test})

    return y_pred

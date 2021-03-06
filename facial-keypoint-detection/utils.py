from datetime import datetime
import os
import time

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import tensorflow as tf

KEY_POINT_NAMES = [
    "left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y",
    "left_eye_inner_corner_x", "left_eye_inner_corner_y", "left_eye_outer_corner_x", "left_eye_outer_corner_y",
    "right_eye_inner_corner_x", "right_eye_inner_corner_y", "right_eye_outer_corner_x", "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y", "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y", "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_left_corner_x", "mouth_left_corner_y", "mouth_right_corner_x", "mouth_right_corner_y",
    "mouth_center_top_lip_x", "mouth_center_top_lip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"
]

ID_LOOKUP_TABLE_PATH = "submissions/IdLookupTable.csv"


class MiniBatchProvider(object):
    def __init__(self, batch_size, ratio_val):
        x, y = load()
        num_val = int(x.shape[0] * ratio_val)

        x_train = x[num_val:, :]
        y_train = y[num_val:, :]
        self.x_train, self.y_train = shuffle(x_train, y_train, random_state=42)
        self.x_val, self.y_val = x[:num_val, :], y[:num_val, :]

        self.batch_size = batch_size
        self.epoch_idx = 0
        self.batch_idx = 0
        self.num_batch = int(x_train.shape[0] / batch_size) + 1

    def prepare(self):
        curr_array_idx = self.batch_size * self.batch_idx
        next_array_idx = self.batch_size * (self.batch_idx + 1)
        x_mini_batch = self.x_train[curr_array_idx:next_array_idx, :]
        y_mini_batch = self.y_train[curr_array_idx:next_array_idx, :]

        if next_array_idx >= self.x_train.shape[0]:
            # epoch 한바퀴 돌았으니 리셋하고 다시 shuffle
            self.batch_idx = 0
            self.epoch_idx += 1
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train, random_state=42)
        else:
            # 다음 mini batch 를 준비
            self.batch_idx += 1

        return x_mini_batch, y_mini_batch, self.x_val, self.y_val

    def get_num_batch(self):
        return self.num_batch


def load(test=False, cols=None, verbose=False):
    df = read_csv("test.csv" if test else "training.csv")

    df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ["Image"]]
    if verbose:
        print(df.count())

    X = np.vstack(df["Image"].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
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


def get_fc():
    tf.reset_default_graph()

    input_images = tf.placeholder(tf.float32, shape=(None, 9216))
    w1 = tf.get_variable("w1", shape=[9216, 100], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([100]))
    hidden = tf.nn.relu(tf.matmul(input_images, w1) + b1)
    w2 = tf.get_variable("w2", shape=[100, 30], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([30]))
    predict = tf.matmul(hidden, w2) + b2

    return input_images, predict


def get_cnn():
    tf.reset_default_graph()

    input_images = tf.placeholder(tf.float32, shape=(None, 9216))
    input_layer = tf.reshape(input_images, [-1, 96, 96, 1])
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

    return input_images, predict


def get_exactly_same_fc_with_blog():
    tf.reset_default_graph()

    input_images = tf.placeholder(tf.float32, shape=(None, 9216))
    w1 = tf.get_variable("w1", shape=[9216, 100], initializer=tf.keras.initializers.glorot_uniform())
    b1 = tf.Variable(tf.zeros([100]), dtype=tf.float32)
    hidden = tf.nn.relu(tf.matmul(input_images, w1) + b1)
    w2 = tf.get_variable("w2", shape=[100, 30], initializer=tf.keras.initializers.glorot_uniform())
    b2 = tf.Variable(tf.zeros([30]), dtype=tf.float32)
    predict = tf.matmul(hidden, w2) + b2

    return input_images, predict


def train(num_epoch, name, model_fn, batch_size=64, ratio_val=0.2, learning_rate=0.01, save_every=100):
    input_images, predict = model_fn()
    truth = tf.placeholder(tf.float32, shape=(None, 30))
    mask = tf.logical_not(tf.is_nan(truth))

    loss = tf.losses.mean_squared_error(tf.boolean_mask(truth, mask),
                                        tf.boolean_mask(predict, mask))
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss)

    provider = MiniBatchProvider(batch_size=batch_size, ratio_val=ratio_val)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_summary = tf.summary.scalar("loss", loss)

        datetime_now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        train_summary_path = "{}_{}_train".format(datetime_now, name)
        val_summary_path = "{}_{}_val".format(datetime_now, name)
        train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", train_summary_path))
        val_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "summaries", val_summary_path))

        for e in range(1, num_epoch + 1):
            start_ms = int(time.time() * 1000)

            for b in range(provider.get_num_batch()):
                x_train, y_train, x_val, y_val = provider.prepare()
                feed_dict_train = {input_images: x_train, truth: y_train}
                feed_dict_val = {input_images: x_val, truth: y_val}
                sess.run(optimizer, feed_dict=feed_dict_train)

            elapsed_ms = int(time.time() * 1000) - start_ms
            train_loss_summary, train_loss = sess.run([loss_summary, loss], feed_dict=feed_dict_train)
            val_loss_summary = sess.run(loss_summary, feed_dict=feed_dict_val)
            print("Epoch: {:4d}\tTraining Loss: {:.7f}\tElapsed Time(ms): {:6d}".format(e, train_loss, elapsed_ms))
            train_writer.add_summary(train_loss_summary, global_step=e)
            val_writer.add_summary(val_loss_summary, global_step=e)
            train_writer.flush()
            val_writer.flush()

            if e % save_every == 0 or e == num_epoch:
                save_dir = "checkpoints/{}_{}/{:05d}".format(datetime_now, name, e)
                os.makedirs(save_dir)
                save_path = "{}/ckpt".format(save_dir)
                print("model has saved in path: {}".format(saver.save(sess, save_path)))

        X_test, _ = load(test=True)
        y_pred = sess.run(predict, feed_dict={input_images: X_test})

    return y_pred


def prepare_submission(dir_name, target_epoch, model_fn):
    X_test, _ = load(test=True)
    ckpt_path = os.path.join(os.getcwd(), "checkpoints", dir_name, "{:05d}".format(target_epoch), "ckpt")

    input_images, predict = model_fn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        out = sess.run(predict, feed_dict={input_images: X_test})
    v_scale_up = np.vectorize(lambda v: min(max(v, -1), 1) * 48 + 48)
    pixels = v_scale_up(out)

    submission_csv_path = os.path.join(os.getcwd(), "submissions", "{}_{:05d}.csv".format(dir_name, target_epoch))
    prepare_submission_slave(pixels, submission_csv_path)

    return submission_csv_path


def prepare_submission_slave(pixels, csv_path):
    df_image_id = pd.DataFrame(data=list(range(1, pixels.shape[0] + 1)), columns=["ImageId"])
    df_pixels = pd.DataFrame(data=pixels, columns=KEY_POINT_NAMES)
    df_key_points_pivot = pd.concat([df_image_id, df_pixels], axis=1)
    df_key_points = pd.melt(df_key_points_pivot,
                            id_vars=["ImageId"], value_vars=KEY_POINT_NAMES,
                            var_name="FeatureName", value_name="Location")
    df_key_points = df_key_points.sort_values(by=["ImageId"])
    df_id_lookup = read_csv(ID_LOOKUP_TABLE_PATH)

    submission_df = pd.merge(df_id_lookup, df_key_points, on=["ImageId", "FeatureName"], suffixes=["_", ""])
    submission_df[["RowId", "Location"]].to_csv(csv_path, index=False)

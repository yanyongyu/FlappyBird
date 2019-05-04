# -*- coding: utf-8 -*-
"""
This is the DQN module of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"

import random
from collections import deque
import logging

import cv2 as cv
import numpy as np
import tensorflow as tf

import main

logging.basicConfig(level=logging.INFO)

GAME = "FlappyBird"
# 动作数量
ACTIONS = 2
FRAME_PER_ACTION = 1
# 历史观察奖励衰减
GAMMA = 0.99
# 训练前观察积累论述
OBSERVE = 100000.
# frames over which to anneal epsilon
EXPLORE = 2000000.
# final value of epsilon
FINAL_EPSILON = 0.0001
# starting value of epsilon
INITIAL_EPSILON = 0.1
# number of previous transitions to remember
REPLAY_MEMORY = 50000
# size of minibatch
BATCH = 32


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# 神经网络结构
def createNetwork():
    # 输入层
    s = tf.placeholder(tf.float32, [None, 80, 80, 4])

    # 卷积层1
    # w_conv1 = weight_variable([3, 3, 4, 16])
    # b_conv1 = bias_variable([16])
    # conv1 = tf.nn.relu(conv2d(s, w_conv1) + b_conv1)
    w_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(s, w_conv1, 4) + b_conv1)
    pool1 = max_pool_2x2(conv1)

    # 卷积层2
    # w_conv2_1 = weight_variable([3, 3, 16, 32])
    # b_conv2_1 = weight_variable([32])
    # conv2_1 = tf.nn.relu(conv2d(pool1, w_conv2_1) + b_conv2_1)
    w_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = weight_variable([64])
    conv2 = tf.nn.relu(conv2d(pool1, w_conv2, 2) + b_conv2)

    # w_conv2_2 = weight_variable([3, 3, 32, 32])
    # b_conv2_2 = weight_variable([32])
    # conv2_2 = tf.nn.relu(conv2d(conv2_1, w_conv2_2) + b_conv2_2)
    # pool2 = max_pool_2x2(conv2_2)

    # 卷积层3
    # w_conv3_1 = weight_variable([3, 3, 32, 64])
    # b_conv3_1 = bias_variable([64])
    # conv3_1 = tf.nn.relu(conv2d(pool2, w_conv3_1) + b_conv3_1)

    # w_conv3_2 = weight_variable([3, 3, 64, 64])
    # b_conv3_2 = bias_variable([64])
    # conv3_2 = tf.nn.relu(conv2d(conv3_1, w_conv3_2) + b_conv3_2)

    # w_conv3_3 = weight_variable([3, 3, 64, 64])
    # b_conv3_3 = bias_variable([64])
    # conv3_3 = tf.nn.relu(conv2d(conv3_2, w_conv3_3) + b_conv3_3)
    # pool3 = max_pool_2x2(conv3_3)
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    conv3 = tf.nn.relu(conv2d(conv2, w_conv3) + b_conv3)

    # 拉直
    # flat = tf.reshape(pool3, [-1, 6400])
    flat = tf.reshape(conv3, [-1, 1600])

    # 全连接层1
    w_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)

    # 全连接层2
    w_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    readout = tf.matmul(fc1, w_fc2) + b_fc2

    return s, readout


def trainNetwork(s, readout, sess, isTrain=False):
    # 定义loss
    a = tf.placeholder(tf.float32, [None, ACTIONS])
    y = tf.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a),
                                   reduction_indices=1)
    loss = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    # 连接游戏模拟器
    game = main.Game()
    game.init_vars(ai=True)

    # 初始化双向队列
    D = deque()

    # 初始化状态，预处理图片
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, reward_0, terminal = game.intelligence(do_nothing)
    x_t = cv.cvtColor(cv.resize(x_t, (80, 80)), cv.COLOR_BGR2GRAY)
    ret, x_t = cv.threshold(x_t, 1, 255, cv.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 加载神经网络参数
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("model")
    step = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        step = int(checkpoint.model_checkpoint_path.split('-')[-1])
        logging.info("Successfully loaded: %s"
                     % checkpoint.model_checkpoint_path)
    else:
        logging.info("Could not find old network weights")

    # 训练神经网络
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # 根据输入s_t选择动作a_t
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() < epsilon and isTrain:
                logging.info("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        else:
            a_t[0] = 1

        # 逐渐降低epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 更新一轮模拟器图像，奖励值，是否死亡
        x_t1, reward_t1, terminal = game.intelligence(a_t)
        x_t1 = cv.cvtColor(cv.resize(x_t1, (80, 80)), cv.COLOR_BGR2GRAY)
        ret, x_t1 = cv.threshold(x_t1, 1, 255, cv.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 将结果存入双向队列D
        D.append((s_t, a_t, reward_t1, s_t1, terminal))
        # 如果队列已满则弹出左侧最早的结果
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE and isTrain:
            if t == OBSERVE + 1:
                if step:
                    t = step
                epsilon = (INITIAL_EPSILON
                           - (INITIAL_EPSILON
                              - FINAL_EPSILON)
                           * (t - OBSERVE)
                           / EXPLORE)
            # 随机抽取BATCH个样本
            minibatch = random.sample(D, BATCH)
            # 重组样本
            s_t_batch = [d[0] for d in minibatch]
            a_t_batch = [d[1] for d in minibatch]
            reward_t1_batch = [d[2] for d in minibatch]
            s_t1_batch = [d[3] for d in minibatch]

            # 组合reward
            y_batch = []
            readout_t1_batch = readout.eval(feed_dict={s: s_t1_batch})
            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(reward_t1_batch[i])
                else:
                    y_batch.append(reward_t1_batch[i]
                                   + GAMMA * np.max(readout_t1_batch[i]))

            # 反向训练
            train_step.run(feed_dict={
                    y: y_batch,
                    a: a_t_batch,
                    s: s_t_batch
                })

        # 更新状态
        s_t = s_t1
        t += 1

        # 每一千轮保存一次参数
        if t % 1000 == 0 and t > OBSERVE and isTrain:
            saver.save(sess, "model/"+GAME+"-dqn", global_step=t)

        # 打印信息
        logging.info("TIMESTAMP %s | ACTION %s | REWARD %s | Q_max %e"
                     %
                     (t, action_index, reward_t1, np.max(readout_t)))


def play():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.InteractiveSession(
            config=tf.ConfigProto(gpu_options=gpu_options))
    s, readout = createNetwork()
    trainNetwork(s, readout, sess, True)


if __name__ == "__main__":
    play()

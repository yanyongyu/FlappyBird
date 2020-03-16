# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:56:48 2019
Author: yan
"""
__author__ = "yanyongyu"

import random
import logging
from collections import deque

import cv2 as cv
import numpy as np
import tensorflow as tf

import main

logging.basicConfig(level=logging.INFO)

GAME = "FlappyBird"
MODEL_SAVE_PATH = "dqn_model"
LOG_SAVE_PATH = "dqn_logs"
# 动作数量
ACTIONS = 2
FRAME_PER_ACTION = 1
# 历史观察奖励衰减
GAMMA = 0.99
# 训练前观察积累论述
OBSERVE = 50000.
# frames over which to anneal epsilon
EXPLORE = 2000000.
# final value of epsilon
FINAL_EPSILON = 0.
# starting value of epsilon
INITIAL_EPSILON = 0.03
# number of previous transitions to remember
REPLAY_MEMORY = 50000
# size of minibatch
BATCH = 32


class DQN(object):

    def __init__(self, isTrain=False):
        self.isTrain = isTrain
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.replayMemory = deque()
        self.create_network()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            gpu_options=gpu_options))

        self.load_saved_network()

    def weight_variable(self, shape, *, trainable=True) -> tf.Variable:
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, trainable=trainable)

    def bias_variable(self, shape, *, trainable=True) -> tf.Variable:
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, trainable=trainable)

    def conv2d(self, x, W, stride=1):
        return tf.nn.conv2d(x,
                            W,
                            strides=[1, stride, stride, 1],
                            padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

    def set_initial_state(self):
        self.game = main.Game()
        self.game.init_vars(ai=True)

        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, reward_0, terminal = self.game.intelligence(do_nothing)
        x_t = cv.cvtColor(cv.resize(x_t, (80, 80)), cv.COLOR_BGR2GRAY)
        ret, x_t = cv.threshold(x_t, 1, 255, cv.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        return s_t

    def update_state(self, state):
        state_next = cv.cvtColor(cv.resize(state, (80, 80)), cv.COLOR_BGR2GRAY)
        ret, state_next = cv.threshold(state_next, 1, 255, cv.THRESH_BINARY)
        state_next = np.reshape(state_next, (80, 80, 1))
        state_next = np.append(state_next, self.currentState[:, :, :3], axis=2)
        return state_next

    def load_saved_network(self):
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        self.step = 0
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            logging.info("Successfully loaded: %s" %
                         checkpoint.model_checkpoint_path)
        else:
            logging.info("Could not find old network weights")

    def getAction(self):
        self.readout = self.net_readout.eval(
            feed_dict={self.net_input: [self.currentState]})[0]
        action = np.zeros(ACTIONS)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon and self.isTrain:
                logging.info("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(self.readout)
        action[action_index] = 1

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action, action_index

    def create_network(self):
        # 输入层
        self.net_input = tf.placeholder("float", [None, 80, 80, 4])

        # 卷积层1
        # w_conv1 = weight_variable([3, 3, 4, 16])
        # b_conv1 = bias_variable([16])
        # conv1 = tf.nn.relu(conv2d(s, w_conv1) + b_conv1)
        w_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])
        conv1 = tf.nn.relu(self.conv2d(self.net_input, w_conv1, 4) + b_conv1)
        pool1 = self.max_pool_2x2(conv1)

        # 卷积层2
        # w_conv2_1 = weight_variable([3, 3, 16, 32])
        # b_conv2_1 = weight_variable([32])
        # conv2_1 = tf.nn.relu(conv2d(pool1, w_conv2_1) + b_conv2_1)
        w_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.weight_variable([64])
        conv2 = tf.nn.relu(self.conv2d(pool1, w_conv2, 2) + b_conv2)

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
        w_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        conv3 = tf.nn.relu(self.conv2d(conv2, w_conv3) + b_conv3)

        # 拉直
        # flat = tf.reshape(pool3, [-1, 6400])
        flat = tf.reshape(conv3, [-1, 1600])

        # 全连接层1
        w_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
        fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)

        # 全连接层2
        w_fc2 = self.weight_variable([512, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])
        self.net_readout = tf.matmul(fc1, w_fc2) + b_fc2

        # build train network
        self.action_input = tf.placeholder("float", [None, ACTIONS])
        self.q_target = tf.placeholder("float", [None])

        # readout_action -- reward of selected action by a.
        self.q_eval = tf.reduce_sum(tf.multiply(self.net_readout,
                                                self.action_input),
                                    axis=1)
        self.cost = tf.reduce_mean(tf.square(self.q_target - self.q_eval))
        tf.summary.scalar('loss', self.cost)
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def train_network(self):
        # Train the network
        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step2: calculate q_target
        q_target = []
        selected_q_next = self.net_readout.eval(
            feed_dict={self.net_input: next_state_batch})

        for i in range(BATCH):
            terminal = minibatch[i][4]
            if terminal:
                q_target.append(reward_batch[i])
            else:
                q_target.append(reward_batch[i] + GAMMA * selected_q_next[i])

        _, result = self.sess.run(
            [self.train_step, self.merged],
            feed_dict={
                self.q_target: q_target,
                self.action_input: action_batch,
                self.eval_net_input: state_batch
            })
        if (self.timeStep + 1) % 1000 == 0:
            self.writer.add_summary(result, global_step=self.timeStep + 1)

    def start_train(self):
        self.currentState = self.set_initial_state()
        self.writer = tf.summary.FileWriter(LOG_SAVE_PATH, self.sess.graph)
        self.merged = tf.summary.merge_all()
        while True:
            action, action_index = self.getAction()

            state_next, reward, terminal = self.game.intelligence(action)
            state_next = self.update_state(state_next)

            self.replayMemory.append(
                (self.currentState, action, reward, state_next, terminal))
            if len(self.replayMemory) > REPLAY_MEMORY:
                self.replayMemory.popleft()

            if self.timeStep > OBSERVE and self.isTrain:
                if self.timeStep == OBSERVE + 1 and self.step:
                    self.timeStep = self.step
                    self.epsilon = INITIAL_EPSILON\
                        - (INITIAL_EPSILON - FINAL_EPSILON)\
                        * (self.timeStep - OBSERVE)\
                        / EXPLORE

                self.train_network()

            self.currentState = state_next
            self.timeStep += 1

            # save network every 1,000 iteration
            if self.timeStep % 1000 == 0\
                    and self.timeStep > OBSERVE and self.isTrain:
                self.saver.save(self.sess,
                                f"{MODEL_SAVE_PATH}/{GAME}-dqn",
                                global_step=self.timeStep)

            # 打印信息
            logging.info(f"TIMESTAMP {self.timeStep} "
                         f"| ACTION {action_index} "
                         f"| REWARD {reward} "
                         f"| Q_max {np.max(self.readout):e}")


if __name__ == "__main__":
    dqn = DQN(isTrain=True)
    dqn.start_train()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the setting module of the game.
@Author: yanyongyu
"""
__author__ = "yanyongyu"
__all__ = ["Customize_bird", "Setting_line", "write_json", "read_config"]

import os
import json

import cv2 as cv
import numpy as np
import pygame


class Customize_bird():
    """Using opencv to customize the color of the bird's body and mouth."""

    def __init__(self):
        # 读取红色小鸟图片
        self.images = [
            cv.imread("assets/images/birds/bird2_0.png", cv.IMREAD_UNCHANGED),
            cv.imread("assets/images/birds/bird2_1.png", cv.IMREAD_UNCHANGED),
            cv.imread("assets/images/birds/bird2_2.png", cv.IMREAD_UNCHANGED)
        ]
        # 转换图片为hsv格式，方便提取mask
        hsv = list(map(lambda x: cv.cvtColor(x, cv.COLOR_BGR2HSV), self.images))

        # 提取身体部分mask
        lower_hsv_body = np.array([0, 43, 46])
        upper_hsv_body = np.array([15, 255, 255])
        self.mask_body = [
            cv.inRange(hsv[0], lowerb=lower_hsv_body, upperb=upper_hsv_body),
            cv.inRange(hsv[1], lowerb=lower_hsv_body, upperb=upper_hsv_body),
            cv.inRange(hsv[2], lowerb=lower_hsv_body, upperb=upper_hsv_body)
        ]

        # 提取嘴部mask
        lower_hsv_mouth = np.array([16, 43, 46])
        upper_hsv_mouth = np.array([20, 255, 255])
        self.mask_mouth = [
            cv.inRange(hsv[0], lowerb=lower_hsv_mouth, upperb=upper_hsv_mouth),
            cv.inRange(hsv[1], lowerb=lower_hsv_mouth, upperb=upper_hsv_mouth),
            cv.inRange(hsv[2], lowerb=lower_hsv_mouth, upperb=upper_hsv_mouth)
        ]

        # 提取其他部分mask
        mask = list(map(cv.bitwise_or, self.mask_body, self.mask_mouth))
        mask = list(map(lambda x: cv.bitwise_not(x, x), mask))

        # 抠除身体和嘴部
        self.res = list(
            map(
                lambda i: cv.bitwise_and(
                    self.images[i], self.images[i], mask=mask[i]), [0, 1, 2]))

    def seperate(self, body_color, mouth_color):
        for i in range(3):
            # 生成身体部分颜色
            body = np.zeros(self.images[i].shape, np.uint8)
            body[:, :, 0] = body_color[2]
            body[:, :, 1] = body_color[1]
            body[:, :, 2] = body_color[0]
            body[:, :, 3] = body_color[3]  # 透明度
            body = cv.bitwise_and(body, body, mask=self.mask_body[i])

            # 生成嘴部颜色
            mouth = np.zeros(self.images[i].shape, np.uint8)
            mouth[:, :, 0] = mouth_color[2]
            mouth[:, :, 1] = mouth_color[1]
            mouth[:, :, 2] = mouth_color[0]
            mouth[:, :, 3] = mouth_color[3]  # 透明度
            mouth = cv.bitwise_and(mouth, mouth, mask=self.mask_mouth[i])

            # 组合得到新的小鸟并保存
            new_part = cv.bitwise_or(body, mouth)
            new_bird = cv.bitwise_or(self.res[i], new_part)
            cv.imwrite("assets/images/birds/custom_{}.png".format(i), new_bird)


class Setting_line():
    """
    自定义设置条。
    参数：
        屏幕：用于显示
        位置：左上角x，y坐标
        长度：设置条长度
        当前点位置：百分比位置
        颜色：左侧颜色
        高度：设置条高度
    方法：
        display：显示到屏幕上
        set_point：设置点位置
    """

    def __init__(self, screen, rect, lenth, point, color, height=5):
        self.screen = screen

        self.start = rect
        self.end = (rect[0] + lenth, rect[1])
        self.point = (rect[0] + round(lenth * point), rect[1])

        self.lenth = lenth
        self.height = height
        self.color = color

    def display(self):
        pygame.draw.line(self.screen, (0, 0, 0), self.start, self.end,
                         self.height)
        pygame.draw.line(self.screen, self.color, self.start, self.point,
                         self.height)
        pygame.draw.circle(self.screen, (255, 255, 255), self.point, 5)

    def set_point(self, point):
        self.point = (self.start[0] + round(self.lenth * point), self.start[1])


def read_config() -> tuple:
    """Get config from config.json."""
    # 未读取到则写入默认设置
    if "config.json" not in os.listdir(os.getcwd()):
        write_json()
    conf = json.load(open("config.json", "r"))
    return conf.values()


def write_json(bird: int = 3,
               background: int = 2,
               music: int = 50,
               sound: int = 50) -> None:
    """Write config into config.json."""
    conf = {
        'bird': bird,
        'background': background,
        'volume': music,
        'sound': sound
    }
    json.dump(conf, open("config.json", "w"))
